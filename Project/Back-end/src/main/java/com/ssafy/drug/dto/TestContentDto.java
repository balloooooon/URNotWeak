package com.ssafy.drug.dto;

import lombok.*;

import java.util.List;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TestContentDto {
    int testQNo;
    String testQImg;
    List<TestAnswerDto> testAnswers;

    @Override
    public String toString() {
        return "TestContentDto{" +
                "testQNo=" + testQNo +
                ", testQImg='" + testQImg + '\'' +
                ", testAnswers=" + testAnswers +
                '}';
    }
}
