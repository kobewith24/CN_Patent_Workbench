package com.patent.rag.model;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record VerifyResponse(List<Map<String, Object>> claims, double faithfulness) {
}
