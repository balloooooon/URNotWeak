package com.ssafy.drug.repository;

import com.ssafy.drug.model.DrugBaImage;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DrugBaImageRepository extends JpaRepository<DrugBaImage, Integer> {
    public DrugBaImage findByDrugImgId(int id);
    public Long countBy();
}
