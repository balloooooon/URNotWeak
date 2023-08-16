package com.ssafy.drug.model;

import jakarta.persistence.*;
import lombok.*;
import org.antlr.v4.runtime.misc.NotNull;

@Data
@Getter
@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="test_result")
public class TestResult {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name="test_r_id")
    Integer testRId;

    @NotNull
    @Column(name="test_r_score")
    Integer testRScore;

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

    @Column(name="test_r_count")
    Integer testRCount;

    @Builder
    public TestResult(Integer testRId, Integer testRScore, String testRTitle, String testRSubtitle, String testRContent, String testRImg) {
        this.testRId = testRId;
        this.testRScore = testRScore;
        this.testRTitle = testRTitle;
        this.testRSubtitle = testRSubtitle;
        this.testRContent = testRContent;
        this.testRImg = testRImg;
    }

    public void plusCount(){
        this.testRCount+=1;
    }

//    @Builder
//    public TestResult(Integer testRId, Integer testRScore, String testRTitle, String testRSubtitle, String testRContent, String testRImg, Integer testRCount) {
//        this.testRId = testRId;
//        this.testRScore = testRScore;
//        this.testRTitle = testRTitle;
//        this.testRSubtitle = testRSubtitle;
//        this.testRContent = testRContent;
//        this.testRImg = testRImg;
//        this.testRCount = testRCount;
//    }

}
