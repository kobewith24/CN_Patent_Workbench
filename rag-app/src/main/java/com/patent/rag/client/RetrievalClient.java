package com.patent.rag.client;

import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import com.patent.rag.model.Filters;
import com.patent.rag.model.Hit;
import com.patent.rag.model.RetrieveResponse;
import com.patent.rag.model.VerifyRequest;
import com.patent.rag.model.VerifyResponse;

@Component
public class RetrievalClient {

    private final RestClient restClient;

    public RetrievalClient(@Value("${retrieval.base-url}") String baseUrl) {
        this.restClient = RestClient.builder().baseUrl(baseUrl).build();
    }

    public RetrieveResponse retrieve(String query, int k, Filters filters) {
        return restClient.get()
                .uri(builder -> {
                    var b = builder.path("/retrieve_plan")
                            .queryParam("q", query)
                            .queryParam("k", k)
                            .queryParam("candidates", 50);
                    if (notBlank(filters.ipcSection())) {
                        b.queryParam("ipc_section", filters.ipcSection());
                    }
                    if (notBlank(filters.ipcSubclass())) {
                        b.queryParam("ipc_subclass", filters.ipcSubclass());
                    }
                    if (filters.yearFrom() > 0) {
                        b.queryParam("year_from", filters.yearFrom());
                    }
                    if (filters.yearTo() > 0) {
                        b.queryParam("year_to", filters.yearTo());
                    }
                    if (notBlank(filters.patentType())) {
                        b.queryParam("patent_type", filters.patentType());
                    }
                    if (notBlank(filters.city())) {
                        b.queryParam("city", filters.city());
                    }
                    if (notBlank(filters.applicantType())) {
                        b.queryParam("applicant_type", filters.applicantType());
                    }
                    return b.build();
                })
                .retrieve()
                .body(RetrieveResponse.class);
    }

    public VerifyResponse verify(String answer, List<Hit> hits) {
        return restClient.post()
                .uri("/verify")
                .contentType(MediaType.APPLICATION_JSON)
                .body(new VerifyRequest(answer, hits))
                .retrieve()
                .body(VerifyResponse.class);
    }

    private static boolean notBlank(String value) {
        return value != null && !value.isBlank();
    }
}
