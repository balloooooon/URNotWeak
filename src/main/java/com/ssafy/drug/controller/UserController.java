package com.ssafy.drug.controller;

import com.ssafy.drug.model.User;
import com.ssafy.drug.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
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
@RequestMapping("user")
public class UserController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    UserService userService;

    // 모든 사용자 조회
    @Operation(summary = "모든 사용자 조회", description = "사용자의 id, 점수를 가져오게 됩니다.", tags = { "User Controller" })
    @GetMapping(produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<User>> getAllusers(){
        List<User> user = userService.findAll();
        return new ResponseEntity<List<User>>(user, HttpStatus.OK);

    }

    // 사용자 번호 가져오기
    @Operation(summary = "현재 사용자수 조회", description = "현재 사용자의 수를 가져오게 됩니다.", tags = { "User Controller" })
    @GetMapping(value = "/userno", produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<Long> getUserNo(){
        Long userNo = userService.countUser();
        return new ResponseEntity<Long>(userNo, HttpStatus.OK);
    }

    // 최종점수 저장
    @Operation(summary = "사용자 점수 저장", description = "사용자의 점수를 저장하게 됩니다.", tags = { "User Controller" })
    @PostMapping("/result")
    public ResponseEntity<String> postResult(@RequestBody User user){
        logger.info(String.valueOf(user));
        userService.postResult(user);
        return new ResponseEntity<String>("최종점수를 저장했습니다", HttpStatus.OK);
    }
}
