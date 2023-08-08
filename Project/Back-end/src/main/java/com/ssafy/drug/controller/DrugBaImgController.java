package com.ssafy.drug.controller;

import com.ssafy.drug.dto.DrugBaImgDto;
import com.ssafy.drug.dto.SimulThumbnailDto;
import com.ssafy.drug.service.DrugBaImgService;
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
@RequestMapping("drugimg")
public class DrugBaImgController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    private DrugBaImgService drugBaImgService;

    // 마약전후 사진들 가져오기
    @Operation(summary = "마약전후 사진 요청", description = "마약전후 사진 3개를 가져오게 됩니다.", tags = { "Drug Controller" })
    @GetMapping(produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<DrugBaImgDto>> getDrugBaImgs(){
        List<DrugBaImgDto> imgs = drugBaImgService.selectDrugBaImgs();
        return new ResponseEntity<List<DrugBaImgDto>>(imgs, HttpStatus.OK);
    }
}
