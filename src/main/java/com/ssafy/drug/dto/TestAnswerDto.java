package com.ssafy.drug.dto;

import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TestAnswerDto {
    int testAId;
    String testAContent;
    int testAScore;

    @Override
    public String toString() {
        return "TestAnswerDto{" +
                "testAId=" + testAId +
                ", testAContent='" + testAContent + '\'' +
                ", testAScore=" + testAScore +
                '}';
    }
}
