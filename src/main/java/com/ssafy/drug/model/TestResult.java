package com.ssafy.drug.model;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.antlr.v4.runtime.misc.NotNull;

@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="test_result")
public class TestResult {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name="test_r_id")
    int testRId;

    @NotNull
    @Column(name="test_r_score")
    int testRScore;

    @NotNull
    @Column(name="test_r_title")
    String testRTitle;

    @NotNull
    @Column(name="test_r_subtitle")
    String testRSubtitle;

    @NotNull
    @Column(name="test_r_content")
    String testRContent;

    @NotNull
    @Column(name="test_r_img_url")
    String testRImg;

    @Builder
    public TestResult(int testRId, int testRScore, String testRTitle, String testRSubtitle, String testRContent, String testRImg) {
        this.testRId = testRId;
        this.testRScore = testRScore;
        this.testRTitle = testRTitle;
        this.testRSubtitle = testRSubtitle;
        this.testRContent = testRContent;
        this.testRImg = testRImg;
    }
}
