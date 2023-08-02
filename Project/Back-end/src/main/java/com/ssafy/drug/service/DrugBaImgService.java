package com.ssafy.drug.service;

import com.ssafy.drug.dto.DrugBaImgDto;
import com.ssafy.drug.model.DrugBaImage;
import com.ssafy.drug.repository.DrugBaImageRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Service
@Transactional
public class DrugBaImgService {
    @Autowired
    private DrugBaImageRepository drugBaImageRepository;

    public List<DrugBaImgDto>  selectDrugBaImgs(){
        // 마약전후사진 가져오기
        Long imgCo = drugBaImageRepository.countBy();

        // 랜덤으로 3개의 id 가져오기
        Set<Integer> set = new HashSet<>();
        while (set.size() < 3) {
            Double randomNo = Math.random() * imgCo + 1;
            set.add(randomNo.intValue());
        }
        List<Integer> list = new ArrayList<>(set);

        List<DrugBaImgDto> drugBaImgs = new ArrayList<>();
        for(int i=0; i<list.size(); i++) {
            DrugBaImage drugBaImage = drugBaImageRepository.findByDrugImgId(list.get(i));
            drugBaImgs.add(new DrugBaImgDto(drugBaImage.getDrugBeforeImg(),drugBaImage.getDrugAfterImg()));
        }

        return drugBaImgs;
    }
}
