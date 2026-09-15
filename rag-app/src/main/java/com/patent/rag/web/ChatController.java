package com.patent.rag.web;

import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.patent.rag.model.Filters;
import com.patent.rag.service.ChatService;

@RestController
@RequestMapping("/api")
public class ChatController {

    private final ChatService chatService;

    public ChatController(ChatService chatService) {
        this.chatService = chatService;
    }

    @GetMapping("/chat")
    public SseEmitter chat(
            @RequestParam("q") String question,
            @RequestParam(value = "sessionId", defaultValue = "default") String sessionId,
            @RequestParam(value = "ipcSection", defaultValue = "") String ipcSection,
            @RequestParam(value = "ipcSubclass", defaultValue = "") String ipcSubclass,
            @RequestParam(value = "yearFrom", defaultValue = "0") int yearFrom,
            @RequestParam(value = "yearTo", defaultValue = "0") int yearTo,
            @RequestParam(value = "patentType", defaultValue = "") String patentType,
            @RequestParam(value = "city", defaultValue = "") String city,
            @RequestParam(value = "applicantType", defaultValue = "") String applicantType) {
        SseEmitter emitter = new SseEmitter(300_000L);
        Filters filters = new Filters(ipcSection, ipcSubclass, yearFrom, yearTo,
                patentType, city, applicantType);
        chatService.chat(sessionId, question, filters, emitter);
        return emitter;
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of("status", "ok", "service", "rag-app");
    }
}
