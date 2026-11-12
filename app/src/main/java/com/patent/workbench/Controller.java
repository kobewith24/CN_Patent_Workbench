package com.patent.workbench;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.servlet.http.HttpServletRequest;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.*;
import java.util.regex.Pattern;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api")
public class Controller {
  private final Settings settings;
  private final Store store;
  private final Jobs jobs;
  private final RetrievalClient retrieval;
  private final ObjectMapper mapper;
  private static final Pattern OUT_OF_SCOPE =
      Pattern.compile(
          "(?:是否|会不会|判断|认定|构成).{0,8}侵权|侵权(?:风险|判断|分析)|保护范围|法律状态|诉讼|申请人(?:名称|是谁)|新颖性|创造性");

  public Controller(
      Settings settings, Store store, Jobs jobs, RetrievalClient retrieval, ObjectMapper mapper) {
    this.settings = settings;
    this.store = store;
    this.jobs = jobs;
    this.retrieval = retrieval;
    this.mapper = mapper;
  }

  String workspace(HttpServletRequest req) {
    return req.getAttribute("workspace").toString();
  }

  String text(JsonNode body, String key, String fallback, int max) {
    String value = body.path(key).asText(fallback).trim();
    if (value.isBlank() || value.length() > max)
      throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "字段 " + key + " 长度应为 1–" + max);
    return value;
  }

  @GetMapping("/health")
  public Map<String, Object> health() {
    return Map.of("status", "up", "version", settings.version());
  }

  @PostMapping("/workspaces")
  public Object createWorkspace(@RequestBody JsonNode body) throws Exception {
    return store.createWorkspace(text(body, "title", "我的调研工作区", 100));
  }

  @GetMapping("/info")
  public Object info() {
    Map<String, Object> info = new LinkedHashMap<>();
    info.put("version", settings.version());
    info.put("scope", settings.text("corpus_scope"));
    info.put("model", settings.text("model"));
    info.put("candidates", settings.integer("candidates"));
    info.put("metrics", jobs.metrics());
    try {
      info.put("retrieval", retrieval.health());
    } catch (Exception e) {
      info.put("retrieval", Map.of("ready", false, "message", "检索服务尚未就绪"));
    }
    return info;
  }

  @GetMapping("/metrics")
  public Object metrics() {
    return jobs.metrics();
  }

  @GetMapping("/studies")
  public Object studies(HttpServletRequest req) throws Exception {
    return store.studies(workspace(req));
  }

  @PostMapping("/studies")
  public Object createStudy(HttpServletRequest req, @RequestBody JsonNode body) throws Exception {
    return store.createStudy(workspace(req), text(body, "title", "未命名技术调研", 120));
  }

  @GetMapping("/studies/{id}")
  public Object study(HttpServletRequest req, @PathVariable String id) throws Exception {
    return store.study(workspace(req), id);
  }

  @PatchMapping("/studies/{id}")
  public Object rename(HttpServletRequest req, @PathVariable String id, @RequestBody JsonNode body)
      throws Exception {
    store.rename(workspace(req), id, text(body, "title", "未命名技术调研", 120));
    return store.study(workspace(req), id);
  }

  @DeleteMapping("/studies/{id}")
  public Object deleteStudy(HttpServletRequest req, @PathVariable String id) throws Exception {
    store.deleteStudy(workspace(req), id);
    return Map.of("removed", true);
  }

  @PostMapping("/studies/{id}/search")
  public Object search(HttpServletRequest req, @PathVariable String id, @RequestBody JsonNode body)
      throws Exception {
    String query = text(body, "query", "", 1000);
    scope(query);
    JsonNode previous = (JsonNode) store.study(workspace(req), id).get("state");
    JsonNode filters = body.has("filters") ? body.path("filters") : previous.path("filters");
    if (!filters.isObject()) filters = mapper.createObjectNode();
    return jobs.searchAndSave(workspace(req), id, query, filters, query);
  }

  @PutMapping("/studies/{id}/filters")
  public Object filters(HttpServletRequest req, @PathVariable String id, @RequestBody JsonNode body)
      throws Exception {
    JsonNode validated = retrieval.post("/validate_filters", body.path("filters"));
    JsonNode old = (JsonNode) store.study(workspace(req), id).get("state");
    ObjectNode next = old.isObject() ? (ObjectNode) old.deepCopy() : mapper.createObjectNode();
    next.set("filters", validated);
    store.state(workspace(req), id, next);
    return Map.of("saved", true);
  }

  @GetMapping("/studies/{id}/documents")
  public Object documents(HttpServletRequest req, @PathVariable String id) throws Exception {
    return store.documents(workspace(req), id);
  }

  @PutMapping("/studies/{id}/documents")
  public Object saveDocument(
      HttpServletRequest req, @PathVariable String id, @RequestBody JsonNode body)
      throws Exception {
    String doc = text(body, "doc_id", "", 100);
    JsonNode fetched = retrieval.post("/documents", Map.of("doc_ids", List.of(doc)));
    if (fetched.path("documents").isEmpty())
      throw new ResponseStatusException(HttpStatus.NOT_FOUND, "专利不存在");
    String note = body.path("note").asText("");
    if (note.length() > 10000) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "备注过长");
    store.saveDocument(
        workspace(req),
        id,
        doc,
        fetched.path("documents").get(0),
        note,
        body.path("selected").asBoolean(true));
    return Map.of("saved", true);
  }

  @DeleteMapping("/studies/{id}/documents/{doc}")
  public Object removeDocument(
      HttpServletRequest req, @PathVariable String id, @PathVariable String doc) throws Exception {
    store.removeDocument(workspace(req), id, doc);
    return Map.of("removed", true);
  }

  @GetMapping("/documents/{doc}")
  public Object document(@PathVariable String doc) {
    return retrieval.post("/documents", Map.of("doc_ids", List.of(doc)));
  }

  @PostMapping("/studies/{id}/jobs")
  @ResponseStatus(HttpStatus.ACCEPTED)
  public Object submit(HttpServletRequest req, @PathVariable String id, @RequestBody JsonNode body)
      throws Exception {
    String question = text(body, "question", "请概括技术方案及其摘要证据", 1000);
    scope(question);
    List<String> ids = new ArrayList<>();
    if (body.path("doc_ids").isArray())
      for (JsonNode n : body.path("doc_ids")) {
        if (n.asText().length() > 100)
          throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "专利标识过长");
        ids.add(n.asText());
      }
    return jobs.submit(workspace(req), id, body.path("kind").asText("overview"), question, ids);
  }

  void scope(String question) {
    String focus =
        question.replaceAll(
            "(?:不评价|不判断|不分析|不涉及|不讨论|不考虑|无需判断|无需评价|不要判断|不要分析)(?:法律)?(?:保护范围|法律状态|侵权|新颖性|创造性)", "");
    if (OUT_OF_SCOPE.matcher(focus).find())
      throw new ResponseStatusException(
          HttpStatus.UNPROCESSABLE_ENTITY, "当前语料只有摘要，无法作出侵权、保护范围、法律状态或申请人判断。请改为技术方案检索或比较。");
  }

  @GetMapping("/studies/{id}/jobs")
  public Object history(HttpServletRequest req, @PathVariable String id) throws Exception {
    return store.jobs(workspace(req), id);
  }

  @GetMapping("/jobs/{id}")
  public Object job(HttpServletRequest req, @PathVariable String id) throws Exception {
    return store.job(workspace(req), id);
  }

  @GetMapping(value = "/jobs/{id}/events", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  public SseEmitter events(HttpServletRequest req, @PathVariable String id) throws Exception {
    return jobs.events(workspace(req), id);
  }

  @PostMapping("/jobs/{id}/cancel")
  public Object cancel(HttpServletRequest req, @PathVariable String id) throws Exception {
    jobs.cancel(workspace(req), id);
    return Map.of("cancel_requested", true);
  }

  @GetMapping("/studies/{id}/export")
  public ResponseEntity<byte[]> export(
      HttpServletRequest req,
      @PathVariable String id,
      @RequestParam(defaultValue = "md") String format)
      throws Exception {
    String w = workspace(req);
    Map<String, Object> study = store.study(w, id);
    List<Map<String, Object>> docs = store.documents(w, id);
    String content;
    if (format.equals("csv")) {
      StringBuilder out = new StringBuilder("\ufeff申请号,名称,年份,IPC,类型,备注,摘要\r\n");
      for (Map<String, Object> entry : docs) {
        JsonNode d = (JsonNode) entry.get("document");
        out.append(
                String.join(
                    ",",
                    csv(entry.get("doc_id").toString()),
                    csv(d.path("title").asText()),
                    csv(d.path("year").asText()),
                    csv(d.path("ipc_main").asText()),
                    csv(d.path("patent_type").asText()),
                    csv(entry.get("note").toString()),
                    csv(d.path("abstract").asText())))
            .append("\r\n");
      }
      content = out.toString();
    } else if (format.equals("md")) {
      StringBuilder out =
          new StringBuilder(
              "# "
                  + md(study.get("title").toString())
                  + "\n\n导出时间："
                  + Instant.now()
                  + "\n\n"
                  + settings.text("corpus_scope")
                  + "\n\n");
      JsonNode state = (JsonNode) study.get("state");
      out.append("检索问题：")
          .append(md(state.path("last_query").asText()))
          .append("\n\n过滤条件：`")
          .append(state.path("filters").toString())
          .append("`\n\n");
      out.append("## 已保存的候选专利\n\n");
      for (Map<String, Object> entry : docs) {
        JsonNode d = (JsonNode) entry.get("document");
        out.append("### ")
            .append(md(d.path("title").asText()))
            .append("\n\n申请号：")
            .append(entry.get("doc_id"))
            .append("；年份：")
            .append(d.path("year").asText())
            .append("；IPC：")
            .append(d.path("ipc_main").asText())
            .append("\n\n备注：")
            .append(md(entry.get("note").toString()))
            .append("\n\n摘要：")
            .append(md(d.path("abstract").asText()))
            .append("\n\n");
      }
      out.append("## 分析记录\n\n");
      for (Map<String, Object> j : store.jobs(w, id)) {
        JsonNode result = (JsonNode) j.get("result");
        if (!j.get("status").equals("completed") || result.path("claims").isMissingNode()) continue;
        out.append("### ")
            .append(md(result.path("question").asText()))
            .append("\n\n任务：")
            .append(j.get("id"))
            .append("；模型：")
            .append(result.path("versions").path("model").asText())
            .append("；语料版本：")
            .append(result.path("versions").path("corpus").asText())
            .append("\n\n");
        for (JsonNode c : result.path("claims")) {
          out.append("- **")
              .append(md(c.path("dimension").asText()))
              .append("**（")
              .append(c.path("doc_id").asText())
              .append("）：")
              .append(md(c.path("text").asText()))
              .append("\n");
          for (JsonNode e : c.path("evidence"))
            out.append("  - 证据 `")
                .append(e.path("id").asText())
                .append("`：")
                .append(md(e.path("text").asText()))
                .append("\n");
        }
        if (!result.path("review").isEmpty())
          out.append("\n有 ")
              .append(result.path("review").size())
              .append(" 条结论未通过引用或数值检查，未纳入正式结论。\n");
        out.append("\n原文定位与数值检查不等于语义支持判定，请复核证据。\n\n");
      }
      content = out.toString();
    } else throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "仅支持 md 或 csv");
    return ResponseEntity.ok()
        .header(
            HttpHeaders.CONTENT_DISPOSITION,
            "attachment; filename=patent-study-" + id + "." + format)
        .contentType(
            MediaType.parseMediaType(
                format.equals("csv") ? "text/csv;charset=UTF-8" : "text/markdown;charset=UTF-8"))
        .body(content.getBytes(StandardCharsets.UTF_8));
  }

  private String csv(String s) {
    String stripped = s.stripLeading();
    if (!stripped.isEmpty() && "=+@-".indexOf(stripped.charAt(0)) >= 0) s = "'" + s;
    return "\"" + s.replace("\"", "\"\"") + "\"";
  }

  private String md(String s) {
    return s.replace("<", "&lt;").replace(">", "&gt;").replace("\r", "");
  }
}
