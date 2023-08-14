package com.ssafy.drug.controller;

import com.ssafy.drug.dto.ContentDetailDto;
import com.ssafy.drug.dto.ContentDto;
import com.ssafy.drug.service.ContentService;
import io.swagger.v3.oas.annotations.Operation;
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

    // 좋아요 업데이트
    @Operation(summary = "컨텐트(중독, 코카인)의 좋아요 +1 하기", description = "컨텐트(중독, 코카인)의 좋아요가 +1 된다.", tags = { "Content Controller" })
    @PostMapping(value = "/like/{contentId}", produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<String> updateLike(@PathVariable("contentId")  int contentId){
        contentService.updateLike(contentId);
        return new ResponseEntity<String>("좋아요를 업데이트를 완료했습니다", HttpStatus.OK);
    }

    // 해시태그 검색
    @Operation(summary = "컨텐트(중독, 코카인)의 해시태그 기반 검색", description = "컨텐트(중독, 코카인)들을 해시태그 기반으로 검색해서 반환해준다.", tags = { "Content Controller" })
    @GetMapping(value = "/search", produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<ContentDto>> searchTag(@RequestParam String tag){
        List<ContentDto> contents = contentService.searchTag(tag);
        return new ResponseEntity<List<ContentDto>>(contents, HttpStatus.OK);
    }

    // 컨텐츠 이미지 가져오기
    @Operation(summary = "컨텐트(중독, 코카인) 이미지 가져오기", description = "컨텐트(중독, 코카인)들의 이미지들을 가져온다.", tags = { "Content Controller" })
    @GetMapping(value = "/img", produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<ContentDetailDto>> getImgs(){
        List<ContentDetailDto> imgs = contentService.findAll();
        return new ResponseEntity<List<ContentDetailDto>>(imgs, HttpStatus.OK);
    }
}
