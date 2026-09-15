package com.patent.rag.model;

public record Filters(
        String ipcSection,
        String ipcSubclass,
        int yearFrom,
        int yearTo,
        String patentType,
        String city,
        String applicantType
) {
    public static Filters empty() {
        return new Filters("", "", 0, 0, "", "", "");
    }
}
