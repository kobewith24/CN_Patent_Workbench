package com.patent.workbench;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.*;
import jakarta.annotation.PreDestroy;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@Service
public class Jobs {
  private final Store store;
  private final RetrievalClient retrieval;
  private final ModelRunner models;
  private final Settings settings;
  private final ObjectMapper mapper;
  private final ThreadPoolExecutor executor;
  private final ScheduledExecutorService timers = Executors.newSingleThreadScheduledExecutor();
  private final ConcurrentMap<String, Job> active = new ConcurrentHashMap<>();
  private final AtomicLong rejected = new AtomicLong(),
      completed = new AtomicLong(),
      cancelledCount = new AtomicLong(),
      failed = new AtomicLong();

  public Jobs(
      Store store,
      RetrievalClient retrieval,
      ModelRunner models,
      Settings settings,
      ObjectMapper mapper) {
    this.store = store;
    this.retrieval = retrieval;
    this.models = models;
    this.settings = settings;
    this.mapper = mapper;
    int workers = settings.integer("generation_concurrency");
    executor =
        new ThreadPoolExecutor(
            workers,
            workers,
            0,
            TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(settings.integer("queue_capacity")),
            r -> {
              Thread t = new Thread(r, "analysis-worker");
              t.setDaemon(true);
              return t;
            },
            new ThreadPoolExecutor.AbortPolicy());
  }

  class Job implements Runnable {
    final String id = Store.id(), workspace, study, kind, question;
    final List<String> ids;
    final AtomicBoolean cancel = new AtomicBoolean();
    final List<SseEmitter> listeners = new CopyOnWriteArrayList<>();
    final List<Map<String, Object>> events = new ArrayList<>();
    volatile String status = "queued";
    volatile Thread runningThread;
    volatile Map<String, Object> result = new LinkedHashMap<>();
    final long submitted = System.nanoTime();

    Job(String workspace, String study, String kind, String question, List<String> ids) {
      this.workspace = workspace;
      this.study = study;
      this.kind = kind;
      this.question = question;
      this.ids = ids;
    }

    void save() {
      try {
        store.job(id, workspace, study, status, result);
      } catch (Exception e) {
        throw new IllegalStateException("任务保存失败", e);
      }
    }

    synchronized void emit(String event, Object value) {
      Map<String, Object> row = Map.of("event", event, "data", value, "id", events.size() + 1);
      events.add(row);
      for (SseEmitter e : listeners) send(e, row);
    }

    void phase(String phase, String label) {
      status = phase;
      result.put("phase", label);
      save();
      emit("phase", Map.of("status", phase, "label", label));
    }

    @Override
    public void run() {
      synchronized (this) {
        if (cancel.get() || !status.equals("queued")) return;
        queueMillis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - submitted);
        runningThread = Thread.currentThread();
        phase("running", "正在准备证据");
      }
      try {
        if ((System.nanoTime() - submitted)
            > TimeUnit.SECONDS.toNanos(settings.integer("queue_timeout_seconds")))
          throw new TimeoutException("排队超时");
        if (kind.equals("follow_up")) {
          runFollowUp();
          return;
        }
        JsonNode fetched = retrieval.post("/documents", Map.of("doc_ids", ids));
        if (cancel.get()) throw new CancellationException();
        JsonNode documents = fetched.path("documents");
        if (documents.size() != ids.size())
          throw new IllegalArgumentException("部分专利已不在当前索引中，请重新检索");
        result.put("documents", documents);
        save();
        emit("sources", documents);
        ArrayNode context = mapper.createArrayNode();
        for (JsonNode d : documents) {
          ObjectNode item = context.addObject();
          item.set("doc_id", d.path("doc_id"));
          item.set("title", d.path("title"));
          ArrayNode ev = item.putArray("evidence");
          for (JsonNode e : d.path("evidence")) {
            ObjectNode fragment = ev.addObject();
            fragment.set("id", e.path("id"));
            fragment.set("text", e.path("text"));
          }
        }
        phase("running", "正在依据摘要整理技术结论");
        Map<String, Object> generated = models.generate(question, kind, context, cancel::get);
        if (cancel.get()) throw new CancellationException();
        JsonNode draft = (JsonNode) generated.get("draft");
        if (!draft.path("claims").isArray() || draft.path("claims").size() > 30)
          throw new IllegalStateException("模型结论结构不完整或超出数量限制");
        phase("auditing", "正在检查引用、数值和单位");
        JsonNode audit =
            retrieval.post(
                "/audit", Map.of("claims", draft.path("claims"), "documents", documents));
        if (cancel.get()) throw new CancellationException();
        ArrayNode accepted = mapper.createArrayNode(), review = mapper.createArrayNode();
        for (JsonNode c : audit.path("claims")) {
          if (c.path("status").asText().equals("evidence_located")) accepted.add(c);
          else review.add(c);
        }
        result = new LinkedHashMap<>();
        result.put("kind", kind);
        result.put("question", question);
        result.put("claims", accepted);
        result.put("review", review);
        result.put("audit", audit.path("summary"));
        result.put("documents", documents);
        result.put(
            "limitations",
            List.of("分析仅覆盖所选专利摘要，不构成新颖性、侵权或法律状态判断。", "原文定位与数值检查不等于语义支持判定，使用前请复核证据。"));
        result.put("model_notes", draft.path("limitations"));
        result.put("timing", generated.get("timing"));
        result.put(
            "versions",
            Map.of(
                "app",
                settings.version(),
                "model",
                settings.text("model"),
                "corpus",
                retrieval.health().path("manifest").path("version").asText(),
                "config",
                settings.config()));
        result.put("queue_ms", (double) queueMillis);
        result.put("total_ms", (System.nanoTime() - submitted) / 1_000_000.0);
        finish("completed");
        completed.incrementAndGet();
      } catch (CancellationException e) {
        result.put("message", "分析已取消，已保存的候选仍可查看");
        finish("cancelled");
        cancelledCount.incrementAndGet();
      } catch (Exception e) {
        result.put("message", safe(e));
        result.put("question", question);
        result.put("kind", kind);
        if (cancel.get()) {
          result.put("message", "分析已取消，已保存的候选仍可查看");
          finish("cancelled");
          cancelledCount.incrementAndGet();
        } else {
          finish(e instanceof TimeoutException ? "timed_out" : "failed");
          failed.incrementAndGet();
        }
      } finally {
        runningThread = null;
        Thread.interrupted();
      }
    }

    long queueMillis;

    private void runFollowUp() throws Exception {
      JsonNode state = (JsonNode) store.study(workspace, study).get("state");
      if (state.path("last_query").asText().isBlank())
        throw new IllegalArgumentException("尚无上次检索，请先进行一次普通检索");
      phase("running", "正在结合上次检索改写追问");
      Map<String, Object> generated = models.rewrite(question, state, cancel::get);
      String query = ((JsonNode) generated.get("draft")).path("query").asText().trim();
      if (query.isBlank() || query.length() > 1000)
        throw new IllegalStateException("查询改写失败，请输入完整问题后直接检索");
      if (cancel.get()) throw new CancellationException();
      JsonNode found = searchAndSave(workspace, study, query, state.path("filters"), question);
      result = new LinkedHashMap<>();
      result.put("kind", kind);
      result.put("search", found);
      result.put("rewritten_query", query);
      result.put("original_question", question);
      result.put("timing", generated.get("timing"));
      finish("completed");
      completed.incrementAndGet();
    }

    synchronized void finish(String state) {
      if (Set.of("completed", "cancelled", "failed", "timed_out").contains(status)) return;
      status = state;
      result.put("status", state);
      save();
      emit("result", Map.of("id", id, "status", status, "result", result));
      for (SseEmitter e : listeners) e.complete();
      listeners.clear();
      timers.schedule(() -> active.remove(id), 5, TimeUnit.MINUTES);
    }
  }

  private static String safe(Exception e) {
    Throwable t = e;
    while (t.getCause() != null) t = t.getCause();
    String message = t.getMessage();
    return message == null ? "任务失败，请查看本地日志" : message.substring(0, Math.min(600, message.length()));
  }

  public JsonNode searchAndSave(
      String workspace, String study, String query, JsonNode filters, String original)
      throws Exception {
    store.study(workspace, study);
    JsonNode found =
        retrieval.post(
            "/search",
            Map.of(
                "query",
                query,
                "filters",
                filters,
                "k",
                settings.integer("top_k"),
                "candidates",
                settings.integer("candidates")));
    ObjectNode state = mapper.createObjectNode();
    state.put("last_query", query);
    state.put("original_question", original);
    state.set("filters", filters);
    state.set("last_search", found);
    state.put("updated", Instant.now().toString());
    store.state(workspace, study, state);
    return found;
  }

  public Map<String, Object> submit(
      String workspace, String study, String kind, String question, List<String> ids)
      throws Exception {
    store.study(workspace, study);
    if (!Set.of("overview", "compare", "follow_up").contains(kind))
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "未知分析类型");
    if (question == null || question.isBlank() || question.length() > 1000)
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "请输入 1–1000 字的问题");
    ids = ids == null ? List.of() : ids.stream().distinct().toList();
    if (!kind.equals("follow_up")
        && (ids.isEmpty() || ids.size() > settings.integer("max_analysis_documents")))
      throw new ResponseStatusException(
          HttpStatus.BAD_REQUEST, "请选择 1–" + settings.integer("max_analysis_documents") + " 件专利");
    if (kind.equals("compare") && (ids.size() < 2 || ids.size() > 4))
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "比较任务请选择 2–4 件专利");
    Job job = new Job(workspace, study, kind, question.trim(), ids);
    job.result.put("question", question);
    job.result.put("kind", kind);
    job.save();
    active.put(job.id, job);
    try {
      executor.execute(job);
    } catch (RejectedExecutionException e) {
      rejected.incrementAndGet();
      job.result.put("message", "队列已满，请稍后重试");
      job.finish("failed");
      throw new ResponseStatusException(HttpStatus.TOO_MANY_REQUESTS, "分析队列已满，请稍后重试");
    }
    timers.schedule(
        () -> {
          synchronized (job) {
            if (job.status.equals("queued")) {
              job.cancel.set(true);
              executor.remove(job);
              job.result.put("message", "排队超过等待上限");
              job.finish("timed_out");
            }
          }
        },
        settings.integer("queue_timeout_seconds"),
        TimeUnit.SECONDS);
    return Map.of("id", job.id, "status", job.status);
  }

  public void cancel(String workspace, String id) throws Exception {
    store.job(workspace, id);
    Job job = active.get(id);
    if (job == null) return;
    job.cancel.set(true);
    Thread worker = job.runningThread;
    if (worker != null) worker.interrupt();
    synchronized (job) {
      if (job.status.equals("queued")) {
        executor.remove(job);
        job.result.put("message", "已取消排队");
        job.finish("cancelled");
        cancelledCount.incrementAndGet();
      }
    }
  }

  public SseEmitter events(String workspace, String id) throws Exception {
    Map<String, Object> stored = store.job(workspace, id);
    SseEmitter emitter = new SseEmitter(180_000L);
    Job job = active.get(id);
    if (job == null) {
      send(emitter, Map.of("event", "result", "id", 1, "data", stored));
      emitter.complete();
      return emitter;
    }
    synchronized (job) {
      for (Map<String, Object> e : job.events) send(emitter, e);
      if (Set.of("completed", "failed", "cancelled", "timed_out").contains(job.status)) {
        emitter.complete();
        return emitter;
      }
      job.listeners.add(emitter);
    }
    Runnable detach =
        () -> {
          job.listeners.remove(emitter);
          if (!Set.of("completed", "failed", "cancelled", "timed_out").contains(job.status)) {
            job.cancel.set(true);
            Thread worker = job.runningThread;
            if (worker != null) worker.interrupt();
          }
        };
    emitter.onCompletion(detach);
    emitter.onTimeout(detach);
    emitter.onError(e -> detach.run());
    ScheduledFuture<?> heartbeat =
        timers.scheduleAtFixedRate(
            () -> {
              try {
                emitter.send(SseEmitter.event().comment("keepalive"));
              } catch (Exception e) {
                detach.run();
              }
            },
            5,
            5,
            TimeUnit.SECONDS);
    emitter.onCompletion(() -> heartbeat.cancel(false));
    return emitter;
  }

  private void send(SseEmitter emitter, Map<String, Object> event) {
    try {
      emitter.send(
          SseEmitter.event()
              .id(event.get("id").toString())
              .name(event.get("event").toString())
              .data(mapper.writeValueAsString(event.get("data"))));
    } catch (Exception e) {
      emitter.completeWithError(e);
    }
  }

  public Map<String, Object> metrics() {
    return Map.of(
        "active",
        executor.getActiveCount(),
        "queued",
        executor.getQueue().size(),
        "queue_capacity",
        settings.integer("queue_capacity"),
        "completed",
        completed.get(),
        "cancelled",
        cancelledCount.get(),
        "failed",
        failed.get(),
        "rejected",
        rejected.get());
  }

  @PreDestroy
  public void shutdown() {
    for (Job j : active.values()) j.cancel.set(true);
    executor.shutdownNow();
    timers.shutdownNow();
  }
}
