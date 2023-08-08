package com.ssafy.drug.model;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.antlr.v4.runtime.misc.NotNull;

@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="test_answer")
public class TestAnswer {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name="test_a_id")
    int testAId;

    @ManyToOne(optional = false)
    @JoinColumn(name="test_q_no")
    @NotNull
    TestQuest testQuest;

    @NotNull
    @Column(name="test_a_content")
    String testAContent;

    @NotNull
    @Column(name="test_score")
    int testScore;

    @Builder
    public TestAnswer(int testAId, TestQuest testQuest, String testAContent, int testScore) {
        this.testAId = testAId;
        this.testQuest = testQuest;
        this.testAContent = testAContent;
        this.testScore = testScore;
    }
}
