package com.ssafy.drug.repository.querydsl;

import com.ssafy.drug.dto.ContentDto;

import java.util.List;

public interface ContentRepositoryCustom {
    public List<ContentDto> selectContents();
}
