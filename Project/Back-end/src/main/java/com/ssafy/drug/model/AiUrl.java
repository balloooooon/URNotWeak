package com.ssafy.drug.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Data
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="ai_url")
public class AiUrl {
    @Id
    @Column(name="id")
    private int id;

    @Column(name="url")
    private String url;
}
