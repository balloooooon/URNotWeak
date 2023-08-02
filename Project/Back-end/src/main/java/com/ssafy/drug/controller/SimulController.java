package com.ssafy.drug.controller;

import com.ssafy.drug.dto.SimulDetailDto;
import com.ssafy.drug.dto.SimulResultDto;
import com.ssafy.drug.dto.SimulThumbnailDto;
import com.ssafy.drug.dto.TestContentDto;
import com.ssafy.drug.model.SimulationDetail;
import com.ssafy.drug.service.SimulService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("simulation")
@Tag(name = "Template", description = "템플릿 API Document")
public class SimulController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    SimulService simulService;

    // 시뮬레이션 썸네일 가져오기
    @Operation(summary = "시뮬레이션 썸네일 요청", description = "시뮬레이션 썸네일 영상 3개를 가져오게 됩니다.", tags = { "Simul Controller" })
    @GetMapping(value = "/thumbnail",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<SimulThumbnailDto>> getThumbnails(){
        List<SimulThumbnailDto> thumbnails = simulService.selectThumbnail();
        return new ResponseEntity<List<SimulThumbnailDto>>(thumbnails, HttpStatus.OK);

    }

    // 시뮬레이션 정보 가져오기
    @Operation(summary = "역할별 시뮬레이션 정보 요청", description = "역할(학생(1), 부부(2), 청년(3))별로 시뮬레이션 정보들을 가져오게 됩니다.", tags = { "Simul Controller" })
    @GetMapping(value = "/{simulNo}",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<SimulDetailDto>> getContents(@PathVariable("simulNo") int simulNo){
        List<SimulDetailDto> contents = simulService.selectAllContents(simulNo);
        return new ResponseEntity<List<SimulDetailDto>>(contents, HttpStatus.OK);

    }

    // 시뮬레이션 결과 가져오기
    @Operation(summary = "시뮬레이션 결과 요청", description = "시뮬레이션 결과(뉴스, 마약전후사진)를 가져오게 됩니다.", tags = { "Simul Controller" })
    @GetMapping(value = "/result/{simulNo}",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<SimulResultDto> getResult(@PathVariable("simulNo") int simulNo){
        SimulResultDto result = simulService.selectResult(simulNo);
        return new ResponseEntity<SimulResultDto>(result, HttpStatus.OK);

    }
}
