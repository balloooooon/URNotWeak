package com.ssafy.drug.dto;

import com.ssafy.drug.model.TestResult;
import lombok.*;

// (현재 사용안함) 추후 QueryDsl 사용할 때 사용하자
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TestResultDto {
    String testRTitle;
    String testRSubtitle;
    String testRContent;
    String testRImg;
    float percent;

    public static TestResultDto from(TestResultInterface tr, float percent){
        return new TestResultDto(tr.getTestRTitle(), tr.getTestRSubtitle(), tr.getTestRContent(), tr.getTestRImg(), percent);
    }
}
