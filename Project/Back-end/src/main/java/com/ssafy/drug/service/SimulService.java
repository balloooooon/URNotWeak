package com.ssafy.drug.service;

import com.ssafy.drug.dto.SimulThumbnailDto;
import com.ssafy.drug.model.SimulationDetail;
import com.ssafy.drug.repository.SimulRepository;
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

    public List<SimulThumbnailDto> selectThumbnail(){
        List<SimulationDetail> thumbnails = simulRepository.findBySimulThumbnail(1);
        List<SimulThumbnailDto> results = new ArrayList<>();
        thumbnails.forEach(e -> {
            results.add(SimulThumbnailDto.from(e));
        });
        return results;
    }
}
