package com.ssafy.drug.dto;

import com.ssafy.drug.model.SimulationDetail;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SimulDetailDto {
    Integer simulOrder;
    Integer simulContentType;
    String simulVideoUrl;
    String simulText;
    String simulAnswer1;
    String simulAnswer2;
    Integer simulShift1;
    Integer simulShift2;
    Integer arType;

    public static SimulDetailDto from(SimulationDetail sd){
        return new SimulDetailDto(sd.getSimulOrder(), sd.getSimulContentType(), sd.getSimulVideoUrl(), sd.getSimulText(),
                sd.getSimulAnswer1(), sd.getSimulAnswer2(), sd.getSimulShift1(), sd.getSimulShift2(), sd.getArType());
    }
}
