package com.ssafy.drug.dto;

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
}
