package com.ssafy.drug.repository;

import com.ssafy.drug.model.Content;
import com.ssafy.drug.model.ContentDetail;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ContentDetailRepository extends JpaRepository<ContentDetail, Integer>{
    public List<ContentDetail> findAll();
}
