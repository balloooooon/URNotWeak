package com.ssafy.drug.dto;

import com.ssafy.drug.model.SimulationDetail;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SimulThumbnailDto {
    int simulNo;
    String url;

    public static SimulThumbnailDto from(SimulationDetail sd){
        return new SimulThumbnailDto(sd.getSimulation().getSimulNo(), sd.getSimulVideoUrl());
    }
}
