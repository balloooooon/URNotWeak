package com.ssafy.drug.repository;

import com.querydsl.core.group.GroupBy;
import com.querydsl.core.group.GroupBy.*;
import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.ssafy.drug.dto.TestAnswerDto;
import com.ssafy.drug.dto.TestContentDto;
import com.ssafy.drug.model.QTestAnswer;
import com.ssafy.drug.model.QTestQuest;
import com.ssafy.drug.model.TestAnswer;
import com.ssafy.drug.model.TestQuest;
import com.ssafy.drug.repository.querydsl.TestRepositoryCustom;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

import static com.querydsl.core.group.GroupBy.groupBy;
import static com.querydsl.core.group.GroupBy.list;

@RequiredArgsConstructor
public class TestRepositoryImpl implements TestRepositoryCustom {
    private final JPAQueryFactory queryFactory;
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Override
    @Transactional
    public List<TestContentDto> selectAllContent() {

        QTestQuest qTestQuest = QTestQuest.testQuest;
        QTestAnswer qTestAnswer = QTestAnswer.testAnswer;

        List<TestContentDto> result = queryFactory
                .from(qTestQuest)
                .join(qTestAnswer).on(qTestAnswer.testQuest.testQNo.eq(qTestQuest.testQNo))
                .transform(groupBy(qTestQuest.testQNo).list(Projections.constructor(TestContentDto.class,
                        qTestQuest.testQNo, qTestQuest.testQImg,
                        list(Projections.constructor(TestAnswerDto.class, qTestAnswer.testAId, qTestAnswer.testAContent, qTestAnswer.testScore)
                        ))));

        return result;
    }
}
