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
@Entity(name="drug_ba_img")
public class DrugBaImage {
    @Id
    @Column(name="drug_img_id")
    private int drugImgId;

    @Column(name="drug_before_img")
    private String drugBeforeImg;

    @Column(name="drug_after_img")
    private String drugAfterImg;
}
