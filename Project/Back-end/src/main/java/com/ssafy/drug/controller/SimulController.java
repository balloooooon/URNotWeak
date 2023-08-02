package com.ssafy.drug.controller;

import com.ssafy.drug.dto.SimulDetailDto;
import com.ssafy.drug.dto.SimulResultDto;
import com.ssafy.drug.dto.SimulThumbnailDto;
import com.ssafy.drug.dto.TestContentDto;
import com.ssafy.drug.model.SimulationDetail;
import com.ssafy.drug.service.SimulService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@CrossOrigin(origins = "*", allowedHeaders = "*")
@RequestMapping("simulation")
public class SimulController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    SimulService simulService;

    // 시뮬레이션 썸네일 가져오기
    @GetMapping(value = "/thumbnail",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<SimulThumbnailDto>> getThumbnails(){
        List<SimulThumbnailDto> thumbnails = simulService.selectThumbnail();
        return new ResponseEntity<List<SimulThumbnailDto>>(thumbnails, HttpStatus.OK);

    }

    // 시뮬레이션 정보 가져오기
    @GetMapping(value = "/{simulNo}",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<SimulDetailDto>> getContents(@PathVariable("simulNo") int simulNo){
        List<SimulDetailDto> contents = simulService.selectAllContents(simulNo);
        return new ResponseEntity<List<SimulDetailDto>>(contents, HttpStatus.OK);

    }

    // 시뮬레이션 결과 가져오기
    @GetMapping(value = "/result/{simulNo}",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<SimulResultDto> getResult(@PathVariable("simulNo") int simulNo){
        SimulResultDto result = simulService.selectResult(simulNo);
        return new ResponseEntity<SimulResultDto>(result, HttpStatus.OK);

    }
}
