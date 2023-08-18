package com.ssafy.drug.dto;


import lombok.*;

import java.util.List;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SimulResultDto {
    String news;
    List<DrugBaImgDto> drugBaImgs;
}
