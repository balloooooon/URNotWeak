package com.ssafy.drug.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.persistence.*;
import lombok.*;

@Data
@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="user")
public class User {
    @Id
    @Column(name="user_id")
    @JsonProperty("user_id")
    private int userId;

    @Column(name="test_final_score")
    @JsonProperty("test_final_score")
    private Integer testFinalScore;

    @Builder
    public User(int userId, int testFinalScore){
        this.userId=userId;
        this.testFinalScore=testFinalScore;
    }

    @Builder
    public User(int testFinalScore){
        this.testFinalScore=testFinalScore;
    }
}
