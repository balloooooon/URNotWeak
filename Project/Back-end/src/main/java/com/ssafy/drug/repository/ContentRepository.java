package com.ssafy.drug.repository;

import com.ssafy.drug.dto.ContentDto;
import com.ssafy.drug.model.AiUrl;
import com.ssafy.drug.model.Content;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ContentRepository extends JpaRepository<Content, Integer> {
    @Query(value = "select tr.testRId as testRId, tr.testRScore as testRScore, tr.testRTitle as testRTitle, tr.testRSubtitle as testRSubtitle, tr.testRContent as testRContent, tr.testRImg as testRImg, tr.testRCount as  testRCount \n"+
            "from test_result tr \n"+
            "where tr.testRScore = (select max(tr2.testRScore) from test_result tr2 where tr2.testRScore<=:score)"
    )
    public List<ContentDto> selectContents();
}
