package com.ssafy.drug.controller;

import com.ssafy.drug.dto.ContentDto;
import com.ssafy.drug.dto.DrugBaImgDto;
import com.ssafy.drug.service.ContentService;
import io.swagger.v3.oas.annotations.Operation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@CrossOrigin(origins = "*", allowedHeaders = "*")
@RequestMapping("content")
public class ContentController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private ContentService contentService;

    // 컨텐트 카드 가져오기
    @Operation(summary = "컨텐트(중독, 코카인) 카드 요청", description = "컨텐트(중독, 코카인)의 설명이 적힌 카드들을 반환합니다.", tags = { "Content Controller" })
    @GetMapping(produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<ContentDto>> getContents(){
        List<ContentDto> contents = contentService.selectContents();
        return new ResponseEntity<List<ContentDto>>(contents, HttpStatus.OK);
    }
}
