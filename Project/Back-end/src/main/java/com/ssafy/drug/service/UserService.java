package com.ssafy.drug.service;

import com.ssafy.drug.dto.TestResultInterface;
import com.ssafy.drug.model.TestResult;
import com.ssafy.drug.model.User;
import com.ssafy.drug.repository.TestRepository;
import com.ssafy.drug.repository.UserRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Transactional
@Service
public class UserService {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private TestRepository testRepository;

    public List<User> findAll(){
        List<User> users = new ArrayList<>();
        userRepository.findAll().forEach(e -> users.add(e));
        return users;
    }

    public Long countUser(){
        return userRepository.countUser();
    }

    public void postResult(User user) {
        // 취약테스트 결과 정보 통계 업데이트(현재 점수 범위 개수 + 1) 진행
        TestResultInterface result = testRepository.selectResultContent(user.getTestFinalScore());
        TestResult tr = testRepository.findById(result.getTestRId()).orElseThrow(() -> new IllegalArgumentException("해당 결과가 없습니다. id="+result.getTestRId()));
        // JPA 의 영속성 컨텍스트 덕분에 entity 객체의 값만 변경하면 자동으로 변경사항 반영함!
        // 따라서 repository.update 를 쓰지 않아도 됨.
        tr.plusCount();
        userRepository.save(user);
    }
}
