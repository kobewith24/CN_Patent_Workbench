package com.patent.workbench;

import com.fasterxml.jackson.databind.*;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.sql.*;
import java.time.Instant;
import java.util.*;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

@Component
public class Store {
  private final String url;
  private final ObjectMapper mapper;

  public Store(Settings settings, ObjectMapper mapper) throws Exception {
    this.mapper = mapper;
    url = "jdbc:sqlite:" + settings.dataDir.resolve("workbench.db");
    try (Connection c = connect();
        Statement st = c.createStatement()) {
      st.execute("PRAGMA journal_mode=WAL");
      st.execute(
          "CREATE TABLE IF NOT EXISTS workspaces(id TEXT PRIMARY KEY,title TEXT NOT NULL,key_hash"
              + " TEXT NOT NULL,created TEXT NOT NULL)");
      st.execute(
          "CREATE TABLE IF NOT EXISTS studies(id TEXT PRIMARY KEY,workspace_id TEXT NOT NULL,title"
              + " TEXT NOT NULL,payload TEXT NOT NULL,created TEXT NOT NULL,updated TEXT NOT NULL,"
              + " FOREIGN KEY(workspace_id) REFERENCES workspaces(id))");
      st.execute("CREATE INDEX IF NOT EXISTS studies_workspace ON studies(workspace_id,updated)");
      st.execute(
          "CREATE TABLE IF NOT EXISTS documents(study_id TEXT NOT NULL,doc_id TEXT NOT"
              + " NULL,document TEXT NOT NULL,note TEXT NOT NULL DEFAULT '',selected INTEGER NOT"
              + " NULL DEFAULT 1,PRIMARY KEY(study_id,doc_id),FOREIGN KEY(study_id) REFERENCES"
              + " studies(id) ON DELETE CASCADE)");
      st.execute(
          "CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,workspace_id TEXT NOT NULL,study_id"
              + " TEXT NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL,created TEXT NOT"
              + " NULL,updated TEXT NOT NULL, FOREIGN KEY(study_id) REFERENCES studies(id) ON"
              + " DELETE CASCADE)");
      st.execute(
          "CREATE INDEX IF NOT EXISTS jobs_workspace ON jobs(workspace_id,study_id,created)");
    }
    // Never silently resume an unfinished generation after a restart.
    try (Connection c = connect();
        Statement st = c.createStatement()) {
      st.executeUpdate(
          "UPDATE jobs SET status='interrupted' WHERE status IN ('queued','running','auditing')");
    }
  }

  private Connection connect() throws SQLException {
    Connection c = DriverManager.getConnection(url);
    try (Statement s = c.createStatement()) {
      s.execute("PRAGMA foreign_keys=ON");
      s.execute("PRAGMA busy_timeout=5000");
    }
    return c;
  }

  static String id() {
    return UUID.randomUUID().toString();
  }

  static String hash(String value) {
    try {
      return HexFormat.of()
          .formatHex(
              MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)));
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
  }

  private String json(Object o) {
    try {
      return mapper.writeValueAsString(o);
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  private JsonNode parse(String s) {
    try {
      return mapper.readTree(s);
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  public Map<String, Object> createWorkspace(String title) throws SQLException {
    String id = id(), key = id() + id(), now = Instant.now().toString();
    try (Connection c = connect();
        PreparedStatement p = c.prepareStatement("INSERT INTO workspaces VALUES(?,?,?,?)")) {
      p.setString(1, id);
      p.setString(2, title);
      p.setString(3, hash(key));
      p.setString(4, now);
      p.executeUpdate();
    }
    return Map.of("id", id, "key", key, "title", title);
  }

  public boolean workspaceValid(String id, String key) throws SQLException {
    if (id == null || key == null) return false;
    try (Connection c = connect();
        PreparedStatement p = c.prepareStatement("SELECT key_hash FROM workspaces WHERE id=?")) {
      p.setString(1, id);
      try (ResultSet r = p.executeQuery()) {
        return r.next()
            && MessageDigest.isEqual(
                hash(key).getBytes(StandardCharsets.US_ASCII),
                r.getString(1).getBytes(StandardCharsets.US_ASCII));
      }
    }
  }

  public Map<String, Object> createStudy(String workspace, String title) throws SQLException {
    String id = id(), now = Instant.now().toString();
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "INSERT INTO studies(id,workspace_id,title,payload,created,updated)"
                    + " VALUES(?,?,?,?,?,?)")) {
      p.setString(1, id);
      p.setString(2, workspace);
      p.setString(3, title);
      p.setString(4, "{}");
      p.setString(5, now);
      p.setString(6, now);
      p.executeUpdate();
    }
    return study(workspace, id);
  }

  public List<Map<String, Object>> studies(String workspace) throws SQLException {
    List<Map<String, Object>> out = new ArrayList<>();
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "SELECT id,title,created,updated FROM studies WHERE workspace_id=? ORDER BY updated"
                    + " DESC LIMIT 200")) {
      p.setString(1, workspace);
      try (ResultSet r = p.executeQuery()) {
        while (r.next())
          out.add(
              Map.of(
                  "id",
                  r.getString(1),
                  "title",
                  r.getString(2),
                  "created",
                  r.getString(3),
                  "updated",
                  r.getString(4)));
      }
    }
    return out;
  }

  public Map<String, Object> study(String workspace, String id) throws SQLException {
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "SELECT title,payload,created,updated FROM studies WHERE workspace_id=? AND"
                    + " id=?")) {
      p.setString(1, workspace);
      p.setString(2, id);
      try (ResultSet r = p.executeQuery()) {
        if (!r.next()) throw new ResponseStatusException(HttpStatus.NOT_FOUND, "调研不存在或无权访问");
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("title", r.getString(1));
        m.put("state", parse(r.getString(2)));
        m.put("created", r.getString(3));
        m.put("updated", r.getString(4));
        return m;
      }
    }
  }

  public void deleteStudy(String workspace, String id) throws SQLException {
    // One atomic statement protects against a queued/running job appearing while deleting.
    // SQLite's existing foreign keys remove only this study's documents and job history.
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "DELETE FROM studies WHERE workspace_id=? AND id=? AND NOT EXISTS"
                    + " (SELECT 1 FROM jobs WHERE study_id=? AND workspace_id=?"
                    + " AND status IN ('queued','running','auditing'))")) {
      p.setString(1, workspace);
      p.setString(2, id);
      p.setString(3, id);
      p.setString(4, workspace);
      if (p.executeUpdate() == 1) return;
    }
    study(workspace, id);
    throw new ResponseStatusException(HttpStatus.CONFLICT, "该调研仍有进行中或排队的分析，请先停止任务再删除。");
  }

  public void state(String workspace, String study, Object payload) throws SQLException {
    study(workspace, study);
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "UPDATE studies SET payload=?,updated=? WHERE id=? AND workspace_id=?")) {
      p.setString(1, json(payload));
      p.setString(2, Instant.now().toString());
      p.setString(3, study);
      p.setString(4, workspace);
      p.executeUpdate();
    }
  }

  public void rename(String workspace, String study, String title) throws SQLException {
    study(workspace, study);
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "UPDATE studies SET title=?,updated=? WHERE id=? AND workspace_id=?")) {
      p.setString(1, title);
      p.setString(2, Instant.now().toString());
      p.setString(3, study);
      p.setString(4, workspace);
      p.executeUpdate();
    }
  }

  public void saveDocument(
      String workspace, String study, String id, Object document, String note, boolean selected)
      throws SQLException {
    study(workspace, study);
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "INSERT INTO documents(study_id,doc_id,document,note,selected) VALUES(?,?,?,?,?) ON"
                    + " CONFLICT(study_id,doc_id) DO UPDATE SET"
                    + " document=excluded.document,note=excluded.note,selected=excluded.selected")) {
      p.setString(1, study);
      p.setString(2, id);
      p.setString(3, json(document));
      p.setString(4, note);
      p.setBoolean(5, selected);
      p.executeUpdate();
    }
  }

  public List<Map<String, Object>> documents(String workspace, String study) throws SQLException {
    study(workspace, study);
    List<Map<String, Object>> out = new ArrayList<>();
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "SELECT doc_id,document,note,selected FROM documents WHERE study_id=? ORDER BY"
                    + " rowid")) {
      p.setString(1, study);
      try (ResultSet r = p.executeQuery()) {
        while (r.next())
          out.add(
              Map.of(
                  "doc_id",
                  r.getString(1),
                  "document",
                  parse(r.getString(2)),
                  "note",
                  r.getString(3),
                  "selected",
                  r.getBoolean(4)));
      }
    }
    return out;
  }

  public void removeDocument(String workspace, String study, String id) throws SQLException {
    study(workspace, study);
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement("DELETE FROM documents WHERE study_id=? AND doc_id=?")) {
      p.setString(1, study);
      p.setString(2, id);
      p.executeUpdate();
    }
  }

  public void job(String id, String workspace, String study, String status, Object payload)
      throws SQLException {
    String now = Instant.now().toString();
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "INSERT INTO jobs(id,workspace_id,study_id,status,payload,created,updated)"
                    + " VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET"
                    + " status=excluded.status,payload=excluded.payload,updated=excluded.updated")) {
      p.setString(1, id);
      p.setString(2, workspace);
      p.setString(3, study);
      p.setString(4, status);
      p.setString(5, json(payload));
      p.setString(6, now);
      p.setString(7, now);
      p.executeUpdate();
    }
  }

  public List<Map<String, Object>> jobs(String workspace, String study) throws SQLException {
    study(workspace, study);
    List<Map<String, Object>> out = new ArrayList<>();
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "SELECT id,status,payload,created,updated FROM jobs WHERE workspace_id=? AND"
                    + " study_id=? ORDER BY created DESC LIMIT 100")) {
      p.setString(1, workspace);
      p.setString(2, study);
      try (ResultSet r = p.executeQuery()) {
        while (r.next()) out.add(jobRow(r));
      }
    }
    return out;
  }

  public Map<String, Object> job(String workspace, String id) throws SQLException {
    try (Connection c = connect();
        PreparedStatement p =
            c.prepareStatement(
                "SELECT id,status,payload,created,updated FROM jobs WHERE workspace_id=? AND"
                    + " id=?")) {
      p.setString(1, workspace);
      p.setString(2, id);
      try (ResultSet r = p.executeQuery()) {
        if (!r.next()) throw new ResponseStatusException(HttpStatus.NOT_FOUND, "任务不存在或无权访问");
        return jobRow(r);
      }
    }
  }

  private Map<String, Object> jobRow(ResultSet r) throws SQLException {
    return Map.of(
        "id",
        r.getString(1),
        "status",
        r.getString(2),
        "result",
        parse(r.getString(3)),
        "created",
        r.getString(4),
        "updated",
        r.getString(5));
  }
}
