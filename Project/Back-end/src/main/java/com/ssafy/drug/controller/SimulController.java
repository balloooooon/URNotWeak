package com.ssafy.drug.controller;

import com.ssafy.drug.dto.SimulThumbnailDto;
import com.ssafy.drug.dto.TestContentDto;
import com.ssafy.drug.service.SimulService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("simulation")
public class SimulController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    SimulService simulService;

    // 시뮬레이션 썸네일 가져오기
    @GetMapping(value = "/thumbnail",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<SimulThumbnailDto>> getContents(){
        List<SimulThumbnailDto> thumbnails = simulService.selectThumbnail();
        return new ResponseEntity<List<SimulThumbnailDto>>(thumbnails, HttpStatus.OK);

    }


}
