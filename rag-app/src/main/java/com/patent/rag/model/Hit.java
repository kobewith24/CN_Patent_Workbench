package com.patent.rag.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record Hit(
        @JsonProperty("rank") int rank,
        @JsonProperty("score") double score,
        @JsonProperty("doc_id") String docId,
        @JsonProperty("title") String title,
        @JsonProperty("abstract") String abstractText,
        @JsonProperty("ipc_main") String ipcMain,
        @JsonProperty("ipc_section") String ipcSection,
        @JsonProperty("ipc_subclass") String ipcSubclass,
        @JsonProperty("year") int year,
        @JsonProperty("patent_type") String patentType,
        @JsonProperty("applicant_type") String applicantType,
        @JsonProperty("city") String city
) {
}
