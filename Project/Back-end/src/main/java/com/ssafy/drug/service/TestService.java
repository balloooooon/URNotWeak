package com.ssafy.drug.service;

import com.ssafy.drug.dto.TestContentDto;
import com.ssafy.drug.dto.TestResultDto;
import com.ssafy.drug.dto.TestResultInterface;
import com.ssafy.drug.repository.TestRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TestService {
    @Autowired
    private TestRepository testRepository;

    public List<TestContentDto> selectAllContent(){
        return testRepository.selectAllContent();
    }

    public TestResultInterface selectResultContent(int score){
        return testRepository.selectResultContent(score);
    }
}
