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
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;

@Controller
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
    public ResponseEntity<TestResultInterface> getResultContent(@RequestParam int score){
        TestResultInterface result = testService.selectResultContent(score);
        return new ResponseEntity<TestResultInterface>(result, HttpStatus.OK);

    }
}
