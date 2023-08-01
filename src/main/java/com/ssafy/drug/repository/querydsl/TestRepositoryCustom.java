package com.ssafy.drug.repository.querydsl;

import com.ssafy.drug.dto.TestContentDto;

import java.util.List;

public interface TestRepositoryCustom {
    List<TestContentDto> selectAllContent();
}
