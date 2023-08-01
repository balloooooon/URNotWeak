package com.ssafy.drug.dto;


import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SimulResultDto {
    String news;
    String drugBeforeImg;
    String drugAfterImg;
}
