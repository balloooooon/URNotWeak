package com.ssafy.drug.repository;

import com.querydsl.core.group.GroupBy;
import com.querydsl.core.group.GroupBy.*;
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
//        List<TestQuest> answers = queryFactory
//                .select(qTestQuest)
//                .from(qTestQuest)
//                .join(qTestQuest.testAnswers, qTestAnswer)
//                .fetch();

//        Map<Long, TestContentDto> resultMap = queryFactory
//                .from(qTestQuest)
//                .join(qTestAnswer).on(qTestAnswer.testQuest.testQNo.eq(qTestQuest.testQNo))
//                .transform(groupBy(qTestQuest.testQNo).as(new TestContentDto(
//                        qTestQuest.testQNo,
//                        list(new TestAnswerDto(qTestAnswer.testAId, qTestAnswer.testAContent, qTestAnswer.testScore))
//
//                )));

        List<?> fetch = queryFactory
                .from(qTestQuest)
                .join(qTestAnswer).on(qTestAnswer.testQuest.testQNo.eq(qTestQuest.testQNo))
                .fetch();

        return null;
    }
}
