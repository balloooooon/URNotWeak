package com.ssafy.drug.repository;

import com.ssafy.drug.dto.ContentDto;
import com.ssafy.drug.model.AiUrl;
import com.ssafy.drug.model.Content;
import com.ssafy.drug.repository.querydsl.ContentRepositoryCustom;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ContentRepository extends JpaRepository<Content, Integer>, ContentRepositoryCustom {
}
