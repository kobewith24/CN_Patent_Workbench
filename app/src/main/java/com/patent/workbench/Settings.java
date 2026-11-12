package com.patent.workbench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.*;
import java.security.SecureRandom;
import java.util.HexFormat;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class Settings {
  private final JsonNode config;
  public final Path dataDir;
  public final String apiToken;
  public final String internalToken;

  public Settings(
      ObjectMapper mapper,
      @Value("${patent.config:config/workbench.json}") String configPath,
      @Value("${patent.state:data}") String statePath)
      throws IOException {
    config =
        mapper.readTree(
            Path.of(System.getenv().getOrDefault("PATENT_CONFIG", configPath)).toFile());
    dataDir = Path.of(System.getenv().getOrDefault("PATENT_STATE_DIR", statePath)).toAbsolutePath();
    Files.createDirectories(dataDir);
    apiToken = token("access.token", "PATENT_API_TOKEN");
    internalToken = token("internal.token", "PATENT_INTERNAL_TOKEN");
  }

  private String token(String file, String variable) throws IOException {
    String value = System.getenv(variable);
    if (value != null && !value.isBlank()) return value;
    Path path = dataDir.resolve(file);
    if (Files.exists(path)) return Files.readString(path).trim();
    byte[] bytes = new byte[32];
    new SecureRandom().nextBytes(bytes);
    value = HexFormat.of().formatHex(bytes);
    Files.writeString(path, value, StandardOpenOption.CREATE_NEW);
    return value;
  }

  public String text(String key) {
    return config.path(key).asText();
  }

  public int integer(String key) {
    return config.path(key).asInt();
  }

  public String version() {
    return "workbench-2.0.0/prompt-2026-11-12-v2";
  }

  public JsonNode config() {
    return config.deepCopy();
  }
}
