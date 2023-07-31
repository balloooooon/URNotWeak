package com.ssafy.drug.repository;

import com.ssafy.drug.model.TestQuest;
import com.ssafy.drug.repository.querydsl.TestRepositoryCustom;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TestRepository extends JpaRepository<TestQuest, Integer>, TestRepositoryCustom{

}
