package com.patent.rag.model;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record RetrieveResponse(
        String query,
        Map<String, Object> filters,
        int scanned,
        @JsonProperty("elapsed_ms") double elapsedMs,
        List<Hit> hits
) {
}
