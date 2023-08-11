package com.ssafy.drug.dto;

import lombok.*;

import java.util.List;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContentDto {
    int contentId;
    String contentType;
    int contentLike;
    List<ContentDetailDto> details;
    List<String> tags;
}
