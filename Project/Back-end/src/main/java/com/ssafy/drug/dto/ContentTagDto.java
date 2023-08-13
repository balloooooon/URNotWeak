package com.ssafy.drug.dto;

import lombok.*;

import java.util.List;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContentTagDto {
    int contentId;
    List<String> contentName;
}
