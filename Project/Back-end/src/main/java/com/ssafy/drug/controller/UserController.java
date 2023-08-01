package com.ssafy.drug.controller;

import com.ssafy.drug.model.User;
import com.ssafy.drug.service.UserService;
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
@RequestMapping("user")
public class UserController {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());

    @Autowired
    UserService userService;

    // 모든 사용자 조회
    @GetMapping(produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<List<User>> getAllusers(){
        List<User> user = userService.findAll();
        return new ResponseEntity<List<User>>(user, HttpStatus.OK);

    }

    // 사용자 번호 가져오기
    @GetMapping(value = "/userno", produces = { MediaType.APPLICATION_JSON_VALUE })
    public ResponseEntity<Long> getUserNo(){
        Long userNo = userService.countUser();
        return new ResponseEntity<Long>(userNo, HttpStatus.OK);
    }

    // 최종점수 저장
    @PostMapping("/result")
    public ResponseEntity<String> postResult(@RequestBody User user){
        logger.info(String.valueOf(user));
        userService.postResult(user);
        return new ResponseEntity<String>("최종점수를 저장했습니다", HttpStatus.OK);
    }
}
