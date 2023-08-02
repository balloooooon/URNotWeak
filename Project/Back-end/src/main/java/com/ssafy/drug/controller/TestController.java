package com.ssafy.drug.controller;

import com.ssafy.drug.dto.TestContentDto;
import com.ssafy.drug.dto.TestResultDto;
import com.ssafy.drug.dto.TestResultInterface;
import com.ssafy.drug.model.User;
import com.ssafy.drug.service.TestService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@CrossOrigin(origins = "*", allowedHeaders = "*")
@RequestMapping("test")
public class TestController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    TestService testService;

    // 취약테스트 내용 모두 가져오기
    @GetMapping(produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<TestContentDto>> getContents(){
        List<TestContentDto> contents = testService.selectAllContent();
        return new ResponseEntity<List<TestContentDto>>(contents, HttpStatus.OK);

    }

    // 취약테스트 결과 내용 가져오기
    @GetMapping(value = "/result",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<TestResultDto> getResultContent(@RequestParam int score){
        TestResultDto result = testService.selectResultContent(score);
        return new ResponseEntity<TestResultDto>(result, HttpStatus.OK);

    }

    // 결과 통계정보 가져오기(점수기반)
    @GetMapping(value = "/statistics",produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<Float> getResultPercent(@RequestParam int score){
        float percent = testService.getResultPercent(score);
        return new ResponseEntity<Float>(percent, HttpStatus.OK);

    }
}
