package com.patent.rag.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.patent.rag.assistant.PatentAssistant;
import com.patent.rag.client.RetrievalClient;
import com.patent.rag.model.Filters;
import com.patent.rag.model.Hit;
import com.patent.rag.model.RetrieveResponse;
import com.patent.rag.model.VerifyResponse;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.TokenStream;

@Service
public class ChatService {

    private final PatentAssistant assistant;
    private final ChatModel chatModel;
    private final ChatMemoryProvider memoryProvider;
    private final RetrievalClient retrievalClient;
    private final ObjectMapper objectMapper;

    public ChatService(PatentAssistant assistant,
                       ChatModel chatModel,
                       ChatMemoryProvider memoryProvider,
                       RetrievalClient retrievalClient,
                       ObjectMapper objectMapper) {
        this.assistant = assistant;
        this.chatModel = chatModel;
        this.memoryProvider = memoryProvider;
        this.retrievalClient = retrievalClient;
        this.objectMapper = objectMapper;
    }

    public void chat(String sessionId, String question, Filters filters, SseEmitter emitter) {
        String traceId = UUID.randomUUID().toString().replace("-", "").substring(0, 12);
        long startedAt = System.currentTimeMillis();
        try {
            String rewritten = rewrite(sessionId, question);
            RetrieveResponse retrieval = retrievalClient.retrieve(rewritten, 5, filters);
            List<Hit> hits = retrieval == null || retrieval.hits() == null
                    ? List.of() : retrieval.hits();
            long retrievalMs = retrieval == null ? 0 : Math.round(retrieval.elapsedMs());

            send(emitter, "sources", hits);
            if (hits.isEmpty()) {
                String text = "资料不足：没有检索到相关专利。";
                send(emitter, "token", text);
                sendDone(emitter, traceId, startedAt, retrievalMs);
                writeTrace(traceId, question, rewritten, filters, retrievalMs, hits, text, null);
                emitter.complete();
                return;
            }

            String context = buildContext(hits);
            StringBuilder answer = new StringBuilder();
            TokenStream stream = assistant.chat(sessionId, context, question);
            stream.onPartialResponse(token -> {
                        answer.append(token);
                        send(emitter, "token", token);
                    })
                    .onCompleteResponse(response -> {
                        VerifyResponse verification = null;
                        try {
                            verification = retrievalClient.verify(answer.toString(), hits);
                        } catch (Exception ex) {
                            send(emitter, "error", "verify failed: " + ex.getMessage());
                        }
                        Map<String, Object> verifyEvent = new LinkedHashMap<>();
                        verifyEvent.put("claims", verification == null ? List.of() : verification.claims());
                        verifyEvent.put("faithfulness", verification == null ? -1.0 : verification.faithfulness());
                        send(emitter, "verify", verifyEvent);
                        sendDone(emitter, traceId, startedAt, retrievalMs);
                        writeTrace(traceId, question, rewritten, filters, retrievalMs,
                                hits, answer.toString(), verification);
                        emitter.complete();
                    })
                    .onError(error -> {
                        send(emitter, "error", String.valueOf(error.getMessage()));
                        emitter.completeWithError(error);
                    })
                    .start();

        } catch (Exception ex) {
            send(emitter, "error", String.valueOf(ex.getMessage()));
            emitter.completeWithError(ex);
        }
    }

    private String rewrite(String sessionId, String question) {
        try {
            ChatMemory memory = memoryProvider.get(sessionId);
            List<ChatMessage> messages = memory.messages();
            if (messages.isEmpty()) {
                return question;
            }
            StringBuilder history = new StringBuilder();
            for (ChatMessage message : messages) {
                if (message instanceof UserMessage user) {
                    history.append("用户：").append(user.singleText()).append('\n');
                } else if (message instanceof AiMessage ai) {
                    history.append("助手：").append(ai.text()).append('\n');
                }
            }
            if (history.isEmpty()) {
                return question;
            }
            String prompt = """
                    你是检索查询改写器。根据对话历史，把当前问题改写成一个独立、完整、可检索的中文问题。
                    只输出改写后的问题，不要解释、不要引号。

                    对话历史：
                    %s
                    当前问题：%s
                    """.formatted(history, question);
            String rewritten = chatModel.chat(prompt).trim();
            return rewritten.isBlank() ? question : rewritten;
        } catch (Exception ex) {
            return question;
        }
    }

    private String buildContext(List<Hit> hits) {
        StringBuilder sb = new StringBuilder();
        for (Hit hit : hits) {
            sb.append('[').append(hit.rank()).append("] ")
                    .append(hit.title()).append("（")
                    .append(hit.year()).append("，")
                    .append(hit.ipcMain()).append("）\n")
                    .append("摘要：").append(hit.abstractText()).append("\n\n");
        }
        return sb.toString();
    }

    private void sendDone(SseEmitter emitter, String traceId, long startedAt, long retrievalMs) {
        Map<String, Object> done = new LinkedHashMap<>();
        done.put("traceId", traceId);
        done.put("retrievalMs", retrievalMs);
        done.put("elapsedMs", System.currentTimeMillis() - startedAt);
        send(emitter, "done", done);
    }

    private void send(SseEmitter emitter, String event, Object data) {
        try {
            String payload;
            if (data instanceof String text) {
                payload = text;
            } else {
                payload = objectMapper.writeValueAsString(data);
            }
            emitter.send(SseEmitter.event().name(event).data(payload));
        } catch (IOException ignored) {
            // client disconnected
        }
    }

    private void writeTrace(String traceId, String question, String rewritten,
                            Filters filters, long retrievalMs, List<Hit> hits,
                            String answer, VerifyResponse verification) {
        try {
            Path dir = Path.of("logs", "trace");
            Files.createDirectories(dir);
            Map<String, Object> trace = new LinkedHashMap<>();
            trace.put("traceId", traceId);
            trace.put("question", question);
            trace.put("rewritten", rewritten);
            trace.put("filters", filters);
            trace.put("retrievalMs", retrievalMs);
            trace.put("hits", hits);
            trace.put("answer", answer);
            trace.put("claims", verification == null ? List.of() : verification.claims());
            trace.put("faithfulness", verification == null ? -1.0 : verification.faithfulness());
            Files.writeString(dir.resolve(traceId + ".json"),
                    objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(trace));
        } catch (Exception ignored) {
            // trace is best-effort
        }
    }
}
