package com.ssafy.drug.service;

import com.ssafy.drug.dto.DrugBaImgDto;
import com.ssafy.drug.dto.SimulDetailDto;
import com.ssafy.drug.dto.SimulResultDto;
import com.ssafy.drug.dto.SimulThumbnailDto;
import com.ssafy.drug.model.DrugBaImage;
import com.ssafy.drug.model.Simulation;
import com.ssafy.drug.model.SimulationDetail;
import com.ssafy.drug.repository.DrugBaImageRepository;
import com.ssafy.drug.repository.SimulRepository;
import com.ssafy.drug.repository.SimulResultRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
@Transactional
public class SimulService {
    @Autowired
    private SimulRepository simulRepository;

    @Autowired
    private SimulResultRepository simulResultRepository;

    @Autowired
    private DrugBaImageRepository drugBaImageRepository;

    public List<SimulThumbnailDto> selectThumbnail(){
        List<SimulationDetail> thumbnails = simulRepository.findBySimulThumbnail(1);
        List<SimulThumbnailDto> results = new ArrayList<>();
        thumbnails.forEach(e -> {
            results.add(SimulThumbnailDto.from(e));
        });
        return results;
    }

    public List<SimulDetailDto> selectAllContents(int simulNo){
        List<SimulationDetail> contents = simulRepository.findBySimulation_SimulNoOrderBySimulOrderAsc(simulNo);
        List<SimulDetailDto> result = new ArrayList<>();
        contents.forEach(e -> {
            result.add(SimulDetailDto.from(e));
        });
        return result;
    }

    public SimulResultDto selectResult(int simulNo){
        Simulation simul = simulResultRepository.findBySimulNo(simulNo);

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
        return new SimulResultDto(simul.getNewsUrl(), drugBaImgs);
    }
}
