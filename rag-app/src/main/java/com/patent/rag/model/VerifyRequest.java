package com.patent.rag.model;

import java.util.List;

public record VerifyRequest(String answer, List<Hit> sources) {
}
