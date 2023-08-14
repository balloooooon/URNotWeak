package com.ssafy.drug.dto;

import com.ssafy.drug.model.ContentDetail;
import com.ssafy.drug.model.SimulationDetail;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContentDetailDto {
    int contentOrder;
    String contentImg;

    public static ContentDetailDto from(ContentDetail sd){
        return new ContentDetailDto(sd.getContentOrder(), sd.getContentImg());
    }
}
