package com.ssafy.drug.service;

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

import java.util.ArrayList;
import java.util.List;

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
        int randomNo = (int) (Math.random() * imgCo) + 1;
        DrugBaImage drugBaImage = drugBaImageRepository.findByDrugImgId(randomNo);

        return new SimulResultDto(simul.getNewsUrl(), drugBaImage.getDrugBeforeImg(),drugBaImage.getDrugAfterImg());
    }
}
