package com.ssafy.drug.repository;

import com.ssafy.drug.dto.TestResultDto;
import com.ssafy.drug.dto.TestResultInterface;
import com.ssafy.drug.model.TestQuest;
import com.ssafy.drug.model.TestResult;
import com.ssafy.drug.repository.querydsl.TestRepositoryCustom;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

@Repository
public interface TestRepository extends JpaRepository<TestResult, Integer>, TestRepositoryCustom{

    @Query(value = "select tr.testRTitle as testRTitle, tr.testRSubtitle as testRSubtitle, tr.testRContent as testRContent, tr.testRImg as testRImg \n"+
            "from test_result tr \n"+
            "where tr.testRScore = (select max(tr2.testRScore) from test_result tr2 where tr2.testRScore<=:score)"
    )
    public TestResultInterface selectResultContent(@Param("score") int score);
}
