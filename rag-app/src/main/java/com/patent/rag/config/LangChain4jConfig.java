package com.patent.rag.config;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.patent.rag.assistant.PatentAssistant;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.ChatMemoryProvider;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaStreamingChatModel;
import dev.langchain4j.service.AiServices;

@Configuration
public class LangChain4jConfig {

    @Bean
    public ChatModel chatModel(@Value("${ollama.base-url}") String baseUrl,
                               @Value("${ollama.model}") String model) {
        return OllamaChatModel.builder()
                .baseUrl(baseUrl)
                .modelName(model)
                .temperature(0.1)
                .timeout(Duration.ofMinutes(3))
                .build();
    }

    @Bean
    public StreamingChatModel streamingChatModel(@Value("${ollama.base-url}") String baseUrl,
                                                 @Value("${ollama.model}") String model) {
        return OllamaStreamingChatModel.builder()
                .baseUrl(baseUrl)
                .modelName(model)
                .temperature(0.1)
                .timeout(Duration.ofMinutes(3))
                .build();
    }

    @Bean
    public ChatMemoryProvider chatMemoryProvider() {
        Map<String, ChatMemory> memories = new ConcurrentHashMap<>();
        return sessionId -> memories.computeIfAbsent(
                String.valueOf(sessionId),
                key -> MessageWindowChatMemory.withMaxMessages(12));
    }

    @Bean
    public PatentAssistant patentAssistant(StreamingChatModel streamingChatModel,
                                           ChatMemoryProvider chatMemoryProvider) {
        return AiServices.builder(PatentAssistant.class)
                .streamingChatModel(streamingChatModel)
                .chatMemoryProvider(chatMemoryProvider)
                .build();
    }
}
