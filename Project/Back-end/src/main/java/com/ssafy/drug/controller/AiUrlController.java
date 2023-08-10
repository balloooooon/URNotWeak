package com.ssafy.drug.controller;

import com.ssafy.drug.service.AiUrlService;
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
@RequestMapping("aiurl")
public class AiUrlController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private AiUrlService aiUrlService;

    // 주기적으로 바뀌는 AIServer 주소 가져오기
    @Operation(summary = "Ai Server 주소 요청", description = "주기적으로 바뀌는 Ai Server 주소를 반환합니다.", tags = { "Ai Url Controller" })
    @GetMapping(produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<String> getDrugBaImgs(){
        String url = aiUrlService.selectUrl(1);
        return new ResponseEntity<String>(url, HttpStatus.OK);
    }
}
