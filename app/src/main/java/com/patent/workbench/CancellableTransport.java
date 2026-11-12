package com.patent.workbench;

import dev.langchain4j.exception.HttpException;
import dev.langchain4j.http.client.*;
import dev.langchain4j.http.client.jdk.JdkHttpClient;
import dev.langchain4j.http.client.jdk.JdkHttpClientBuilder;
import dev.langchain4j.http.client.sse.*;
import java.time.Duration;
import java.util.concurrent.Flow;
import java.util.concurrent.atomic.*;

/**
 * Connects Ollama's callback adapter to the JDK transport's cancellable stream.
 * OllamaStreamingChatModel 1.23 implements callbacks but not the model Flow API. The transport
 * subscription is available before response headers/first token.
 */
public class CancellableTransport implements HttpClient {
  static final ThreadLocal<Control> CURRENT = new ThreadLocal<>();

  static final class Control implements ServerSentEventParsingHandle {
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicReference<Flow.Subscription> subscription = new AtomicReference<>();

    void attach(Flow.Subscription value) {
      subscription.set(value);
      if (cancelled.get()) value.cancel();
      else value.request(Long.MAX_VALUE);
    }

    public void cancel() {
      cancelled.set(true);
      Flow.Subscription s = subscription.get();
      if (s != null) s.cancel();
    }

    public boolean isCancelled() {
      return cancelled.get();
    }
  }

  private final HttpClient delegate;

  CancellableTransport(HttpClient delegate) {
    this.delegate = delegate;
  }

  public SuccessfulHttpResponse execute(HttpRequest request) {
    return delegate.execute(request);
  }

  public void execute(
      HttpRequest request, ServerSentEventParser parser, ServerSentEventListener listener) {
    Control current = CURRENT.get();
    final Control control = current == null ? new Control() : current;
    delegate.stream(request, parser)
        .subscribe(
            new Flow.Subscriber<>() {
              public void onSubscribe(Flow.Subscription s) {
                control.attach(s);
              }

              public void onNext(HttpStreamingEvent event) {
                if (control.isCancelled()) return;
                if (event instanceof HttpResponseReceived opened)
                  listener.onOpen(opened.response());
                else if (event instanceof ServerSentEvent data)
                  listener.onEvent(data, new ServerSentEventContext(control));
                else if (event instanceof HttpErrorResponseReceived error) {
                  control.cancel();
                  listener.onError(
                      new HttpException(
                          error.statusCode(), "Ollama returned HTTP " + error.statusCode()));
                }
              }

              public void onError(Throwable error) {
                listener.onError(error);
              }

              public void onComplete() {
                listener.onClose();
              }
            });
  }

  public static class Builder implements HttpClientBuilder {
    private final JdkHttpClientBuilder delegate =
        JdkHttpClient.builder()
            .httpClientBuilder(
                java.net.http.HttpClient.newBuilder()
                    .version(java.net.http.HttpClient.Version.HTTP_1_1))
            .streamingBackpressure(true);

    public Duration connectTimeout() {
      return delegate.connectTimeout();
    }

    public HttpClientBuilder connectTimeout(Duration value) {
      delegate.connectTimeout(value);
      return this;
    }

    public Duration readTimeout() {
      return delegate.readTimeout();
    }

    public HttpClientBuilder readTimeout(Duration value) {
      delegate.readTimeout(value);
      return this;
    }

    public HttpClient build() {
      return new CancellableTransport(delegate.build());
    }
  }
}
