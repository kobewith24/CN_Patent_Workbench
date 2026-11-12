package com.patent.workbench;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

import com.fasterxml.jackson.databind.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

@SpringBootTest(properties = "patent.config=../config/workbench.json")
@AutoConfigureMockMvc
class WorkbenchTests {
  static final Path state = temp();

  static Path temp() {
    try {
      return Files.createTempDirectory("patent-workbench-test-");
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @DynamicPropertySource
  static void props(DynamicPropertyRegistry registry) {
    registry.add("patent.state", state::toString);
  }

  @Autowired MockMvc mvc;
  @Autowired ObjectMapper mapper;
  @Autowired Settings settings;
  @Autowired Store store;
  @Autowired Jobs jobs;
  @Autowired Controller controller;
  @MockitoBean RetrievalClient retrieval;
  @MockitoBean ModelRunner models;

  Map<String, Object> workspace() throws Exception {
    return store.createWorkspace("测试工作区");
  }

  MockHttpServletRequestBuilder auth(MockHttpServletRequestBuilder request, Map<String, Object> w) {
    return request
        .header("Authorization", "Bearer " + settings.apiToken)
        .header("X-Workspace-ID", w.get("id"))
        .header("X-Workspace-Key", w.get("key"))
        .contentType("application/json");
  }

  String study(Map<String, Object> w) throws Exception {
    return store.createStudy(w.get("id").toString(), "隔离测试").get("id").toString();
  }

  JsonNode doc() throws Exception {
    return mapper.readTree(
        """
{"documents":[{"doc_id":"CN-TEST","title":"测试技术方案","abstract":"采用2个阀门。","year":2020,"ipc_main":"F16K","patent_type":"发明申请","evidence":[{"id":"CN-TEST:hash:0-7","doc_id":"CN-TEST","text":"采用2个阀门。","start":0,"end":7}]}],"missing":[]}
""");
  }

  @BeforeEach
  void setup() throws Exception {
    when(retrieval.health())
        .thenReturn(mapper.readTree("{\"manifest\":{\"version\":\"test\"},\"ready\":true}"));
    when(retrieval.post(eq("/documents"), any())).thenReturn(doc());
  }

  @Test
  void endpointsRequireBothAccessAndWorkspaceCredentials() throws Exception {
    mvc.perform(get("/api/studies")).andExpect(status().isUnauthorized());
    mvc.perform(get("/api/studies").header("Authorization", "Bearer " + settings.apiToken))
        .andExpect(status().isForbidden());
    Map<String, Object> w = workspace();
    mvc.perform(auth(get("/api/studies"), w)).andExpect(status().isOk());
  }

  @Test
  void crossWorkspaceStudyAndJobAreNotReadable() throws Exception {
    var a = workspace();
    var b = workspace();
    String s = study(a);
    store.job("job-test", a.get("id").toString(), s, "completed", Map.of());
    mvc.perform(auth(get("/api/studies/" + s), b)).andExpect(status().isNotFound());
    mvc.perform(auth(get("/api/jobs/job-test"), b)).andExpect(status().isNotFound());
  }

  @Test
  void persistenceAndInterruptedJobsSurviveReopening() throws Exception {
    var w = workspace();
    String s = study(w), wid = w.get("id").toString();
    store.state(wid, s, Map.of("last_query", "运动模糊"));
    store.saveDocument(wid, s, "CN-TEST", doc().path("documents").get(0), "需要复核", true);
    store.job("restart-test", wid, s, "running", Map.of("question", "测试"));
    Store reopened = new Store(settings, mapper);
    assertEquals(
        "运动模糊", ((JsonNode) reopened.study(wid, s).get("state")).path("last_query").asText());
    assertEquals("需要复核", reopened.documents(wid, s).get(0).get("note"));
    assertEquals("interrupted", reopened.job(wid, "restart-test").get("status"));
  }

  @Test
  void canonicalDocumentIsFetchedInsteadOfTrustingClientText() throws Exception {
    var w = workspace();
    String s = study(w);
    mvc.perform(
            auth(put("/api/studies/" + s + "/documents"), w)
                .content(
                    "{\"doc_id\":\"CN-TEST\",\"document\":{\"abstract\":\"伪造\"},\"note\":\"备注\"}"))
        .andExpect(status().isOk());
    JsonNode saved = (JsonNode) store.documents(w.get("id").toString(), s).get(0).get("document");
    assertEquals("采用2个阀门。", saved.path("abstract").asText());
  }

  @Test
  void invalidClaimsNeverEnterOfficialResult() throws Exception {
    when(models.generate(anyString(), anyString(), any(), any()))
        .thenReturn(
            Map.of(
                "draft",
                mapper.readTree(
                    "{\"claims\":[{\"text\":\"有依据\"},{\"text\":\"无依据\"}],\"limitations\":[]}"),
                "timing",
                Map.of("output_tokens", 10)));
    when(retrieval.post(eq("/audit"), any()))
        .thenReturn(
            mapper.readTree(
                "{\"claims\":[{\"status\":\"evidence_located\",\"text\":\"有依据\"},{\"status\":\"invalid_reference\",\"text\":\"无依据\"}],\"summary\":{}}"));
    var w = workspace();
    String s = study(w), wid = w.get("id").toString();
    String id = jobs.submit(wid, s, "overview", "归纳方案", List.of("CN-TEST")).get("id").toString();
    Map<String, Object> job = await(wid, id);
    assertEquals("completed", job.get("status"));
    JsonNode result = (JsonNode) job.get("result");
    assertEquals(1, result.path("claims").size());
    assertEquals(1, result.path("review").size());
  }

  @Test
  void cancelSignalsUpstreamAndReleasesQueue() throws Exception {
    CountDownLatch entered = new CountDownLatch(1);
    when(models.generate(anyString(), anyString(), any(), any()))
        .thenAnswer(
            inv -> {
              entered.countDown();
              BooleanSupplier cancelled = inv.getArgument(3);
              long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
              while (!cancelled.getAsBoolean() && System.nanoTime() < until) Thread.sleep(10);
              throw new CancellationException();
            });
    var w = workspace();
    String s = study(w), wid = w.get("id").toString();
    String id = jobs.submit(wid, s, "overview", "取消测试", List.of("CN-TEST")).get("id").toString();
    assertTrue(entered.await(2, TimeUnit.SECONDS));
    jobs.cancel(wid, id);
    assertEquals("cancelled", await(wid, id).get("status"));
  }

  @Test
  void fullQueueReturns429AndQueuedCancellationRemovesTask() throws Exception {
    CountDownLatch entered = new CountDownLatch(1);
    when(models.generate(anyString(), anyString(), any(), any()))
        .thenAnswer(
            inv -> {
              entered.countDown();
              BooleanSupplier cancelled = inv.getArgument(3);
              while (!cancelled.getAsBoolean()) Thread.sleep(10);
              throw new CancellationException();
            });
    var w = workspace();
    String s = study(w), wid = w.get("id").toString();
    List<String> ids = new ArrayList<>();
    try {
      ids.add(jobs.submit(wid, s, "overview", "满载测试", List.of("CN-TEST")).get("id").toString());
      assertTrue(entered.await(2, TimeUnit.SECONDS));
      for (int i = 0; i < 4; i++)
        ids.add(jobs.submit(wid, s, "overview", "排队测试", List.of("CN-TEST")).get("id").toString());
      var error =
          assertThrows(
              org.springframework.web.server.ResponseStatusException.class,
              () -> jobs.submit(wid, s, "overview", "应拒绝", List.of("CN-TEST")));
      assertEquals(429, error.getStatusCode().value());
      jobs.cancel(wid, ids.get(1));
      assertEquals(3, jobs.metrics().get("queued"));
    } finally {
      for (String id : ids) jobs.cancel(wid, id);
      for (String id : ids) await(wid, id);
    }
  }

  @Test
  void legalScopeAndInvalidCompareCountAreRejected() throws Exception {
    var w = workspace();
    String s = study(w);
    mvc.perform(
            auth(post("/api/studies/" + s + "/jobs"), w)
                .content("{\"question\":\"判断是否侵权\",\"doc_ids\":[\"CN-TEST\"]}"))
        .andExpect(status().isUnprocessableEntity());
    mvc.perform(
            auth(post("/api/studies/" + s + "/jobs"), w)
                .content("{\"question\":\"比较方案\",\"kind\":\"compare\",\"doc_ids\":[\"CN-TEST\"]}"))
        .andExpect(status().isBadRequest());
  }

  @Test
  void explicitExclusionOfLegalJudgementIsAllowed() {
    assertDoesNotThrow(() -> controller.scope("只比较图像处理步骤，不评价法律保护范围。"));
    assertThrows(
        org.springframework.web.server.ResponseStatusException.class,
        () -> controller.scope("不确定是否侵权，请帮我判断是否侵权。"));
  }

  @Test
  void csvNotesCannotStartAnExecutableFormula() throws Exception {
    var w = workspace();
    String s = study(w);
    store.saveDocument(
        w.get("id").toString(), s, "CN-TEST", doc().path("documents").get(0), "  =1+1", true);
    String csv =
        mvc.perform(auth(get("/api/studies/" + s + "/export?format=csv"), w))
            .andExpect(status().isOk())
            .andReturn()
            .getResponse()
            .getContentAsString(java.nio.charset.StandardCharsets.UTF_8);
    assertTrue(csv.contains("\"'  =1+1\""));
  }

  @Test
  void deletingStudyRemovesItsCandidatesAndHistoryOnly() throws Exception {
    var w = workspace();
    String wid = w.get("id").toString(), removed = study(w), retained = study(w);
    store.saveDocument(wid, removed, "CN-TEST", doc().path("documents").get(0), "备注", true);
    String jobId = Store.id();
    store.job(jobId, wid, removed, "completed", Map.of("question", "待删除的分析"));
    mvc.perform(auth(delete("/api/studies/" + removed), w)).andExpect(status().isOk());
    mvc.perform(auth(get("/api/studies/" + removed), w)).andExpect(status().isNotFound());
    mvc.perform(auth(get("/api/jobs/" + jobId), w)).andExpect(status().isNotFound());
    assertEquals(retained, store.study(wid, retained).get("id"));
    try (var c =
            java.sql.DriverManager.getConnection(
                "jdbc:sqlite:" + settings.dataDir.resolve("workbench.db"));
        var p =
            c.prepareStatement(
                "SELECT (SELECT count(*) FROM documents WHERE study_id=?) + (SELECT count(*) FROM"
                    + " jobs WHERE study_id=?)")) {
      p.setString(1, removed);
      p.setString(2, removed);
      try (var rows = p.executeQuery()) {
        assertTrue(rows.next());
        assertEquals(0, rows.getInt(1));
      }
    }
  }

  @Test
  void studyDeletionCannotCrossWorkspaces() throws Exception {
    var owner = workspace();
    var other = workspace();
    String id = study(owner);
    mvc.perform(auth(delete("/api/studies/" + id), other)).andExpect(status().isNotFound());
    assertEquals(id, store.study(owner.get("id").toString(), id).get("id"));
  }

  @Test
  void studyWithActiveOrQueuedAnalysisCannotBeDeleted() throws Exception {
    var w = workspace();
    String wid = w.get("id").toString();
    for (String state : List.of("queued", "running", "auditing")) {
      String id = study(w), jobId = Store.id();
      store.job(jobId, wid, id, state, Map.of());
      mvc.perform(auth(delete("/api/studies/" + id), w)).andExpect(status().isConflict());
      assertEquals(id, store.study(wid, id).get("id"));
      store.job(jobId, wid, id, "cancelled", Map.of());
      mvc.perform(auth(delete("/api/studies/" + id), w)).andExpect(status().isOk());
    }
  }

  Map<String, Object> await(String workspace, String id) throws Exception {
    long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
    Map<String, Object> job;
    do {
      job = store.job(workspace, id);
      if (!Set.of("queued", "running", "auditing").contains(job.get("status"))) return job;
      Thread.sleep(20);
    } while (System.nanoTime() < until);
    throw new AssertionError("Job did not reach terminal state");
  }
}
