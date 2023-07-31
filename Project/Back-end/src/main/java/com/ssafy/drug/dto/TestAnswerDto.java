package com.ssafy.drug.dto;

public class TestAnswerDto {
    int testAId;
    String testAContent;
    int testAScore;

    public TestAnswerDto(int testAId, String testAContent, int testAScore) {
        this.testAId = testAId;
        this.testAContent = testAContent;
        this.testAScore = testAScore;
    }

    public int getTestAId() {
        return testAId;
    }

    public void setTestAId(int testAId) {
        this.testAId = testAId;
    }

    public String getTestAContent() {
        return testAContent;
    }

    public void setTestAContent(String testAContent) {
        this.testAContent = testAContent;
    }

    public int getTestAScore() {
        return testAScore;
    }

    public void setTestAScore(int testAScore) {
        this.testAScore = testAScore;
    }

    @Override
    public String toString() {
        return "TestAnswerDto{" +
                "testAId=" + testAId +
                ", testAContent='" + testAContent + '\'' +
                ", testAScore=" + testAScore +
                '}';
    }
}
