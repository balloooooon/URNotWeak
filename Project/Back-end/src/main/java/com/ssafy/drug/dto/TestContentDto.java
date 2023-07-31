package com.ssafy.drug.dto;

import java.util.List;

public class TestContentDto {
    int testQNo;
    String testQImg;
    List<TestAnswerDto> testAnswers;

    public TestContentDto(int testQNo, String testQImg, List<TestAnswerDto> testAnswers) {
        this.testQNo = testQNo;
        this.testQImg = testQImg;
        this.testAnswers = testAnswers;
    }

    public int getTestQNo() {
        return testQNo;
    }

    public void setTestQNo(int testQNo) {
        this.testQNo = testQNo;
    }

    public String getTestQImg() {
        return testQImg;
    }

    public void setTestQImg(String testQImg) {
        this.testQImg = testQImg;
    }

    public List<TestAnswerDto> getTestAnswers() {
        return testAnswers;
    }

    public void setTestAnswers(List<TestAnswerDto> testAnswers) {
        this.testAnswers = testAnswers;
    }

    @Override
    public String toString() {
        return "TestContentDto{" +
                "testQNo=" + testQNo +
                ", testQImg='" + testQImg + '\'' +
                ", testAnswers=" + testAnswers +
                '}';
    }
}
