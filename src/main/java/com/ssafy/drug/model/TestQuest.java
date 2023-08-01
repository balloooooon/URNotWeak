package com.ssafy.drug.model;

import jakarta.persistence.*;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="test_quest")
public class TestQuest {

    @Id
    @Column(name="test_q_no")
    private int testQNo;

    @Column(name="test_q_img")
    private String testQImg;

    @OneToMany(mappedBy = "testQuest", cascade = CascadeType.ALL)
    private List<TestAnswer> testAnswers = new ArrayList<>();

    @Builder
    public TestQuest(int testQNo, String testQImg){
        this.testQNo=testQNo;
        this.testQImg=testQImg;
    }
}
