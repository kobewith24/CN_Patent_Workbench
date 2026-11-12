package com.patent.workbench;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import dev.langchain4j.http.client.*;
import dev.langchain4j.http.client.sse.*;
import java.util.concurrent.Flow;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;

class CancellableTransportTests {
  @Test
  void cancelBeforeFirstResponseClosesSubscription() {
    HttpClient delegate = mock(HttpClient.class);
    AtomicBoolean closed = new AtomicBoolean();
    Flow.Publisher<HttpStreamingEvent> publisher =
        listener ->
            listener.onSubscribe(
                new Flow.Subscription() {
                  public void request(long n) {}

                  public void cancel() {
                    closed.set(true);
                  }
                });
    when(delegate.stream(any(), any())).thenReturn(publisher);
    var control = new CancellableTransport.Control();
    CancellableTransport.CURRENT.set(control);
    try {
      new CancellableTransport(delegate)
          .execute(
              HttpRequest.builder().method(HttpMethod.GET).url("http://127.0.0.1/test").build(),
              mock(ServerSentEventParser.class),
              mock(ServerSentEventListener.class));
      assertFalse(closed.get());
      control.cancel();
      assertTrue(closed.get());
    } finally {
      CancellableTransport.CURRENT.remove();
    }
  }
}
