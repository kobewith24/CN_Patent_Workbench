package com.patent.workbench;

import com.fasterxml.jackson.databind.*;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

@Component
public class RetrievalClient {
  private final HttpClient client =
      HttpClient.newBuilder()
          .version(HttpClient.Version.HTTP_1_1)
          .connectTimeout(Duration.ofSeconds(5))
          .build();
  private final Settings settings;
  private final ObjectMapper mapper;

  public RetrievalClient(Settings settings, ObjectMapper mapper) {
    this.settings = settings;
    this.mapper = mapper;
  }

  public JsonNode post(String path, Object body) {
    return request(path, body);
  }

  public JsonNode health() {
    return request("/health", null);
  }

  private JsonNode request(String path, Object body) {
    try {
      HttpRequest.Builder b =
          HttpRequest.newBuilder(URI.create(settings.text("retrieval_url") + path))
              .timeout(
                  Duration.ofSeconds(
                      body == null ? 3 : settings.integer("retrieval_timeout_seconds")))
              .header("X-Internal-Token", settings.internalToken)
              .header("Content-Type", "application/json");
      if (body == null) b.GET();
      else b.POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)));
      HttpResponse<String> response = client.send(b.build(), HttpResponse.BodyHandlers.ofString());
      if (response.statusCode() >= 400)
        throw new ResponseStatusException(
            response.statusCode() == 422 ? HttpStatus.BAD_REQUEST : HttpStatus.BAD_GATEWAY,
            "检索服务拒绝请求：" + response.body().substring(0, Math.min(600, response.body().length())));
      return mapper.readTree(response.body());
    } catch (ResponseStatusException e) {
      throw e;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "检索已中断");
    } catch (Exception e) {
      throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "检索服务暂不可用，请确认 Python 服务已启动");
    }
  }
}
