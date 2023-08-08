package com.ssafy.drug.service;

import com.ssafy.drug.dto.TestContentDto;
import com.ssafy.drug.dto.TestResultDto;
import com.ssafy.drug.dto.TestResultInterface;
import com.ssafy.drug.model.TestResult;
import com.ssafy.drug.repository.TestRepository;
import jakarta.persistence.EntityManager;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@Transactional
public class TestService {
    @Autowired
    private TestRepository testRepository;

    public List<TestContentDto> selectAllContent(){
        return testRepository.selectAllContent();
    }

    public TestResultDto selectResultContent(int score){
        TestResultInterface result = testRepository.selectResultContent(score);
        TestResult tr = testRepository.findById(result.getTestRId()).orElseThrow(() -> new IllegalArgumentException("해당 결과가 없습니다. id="+result.getTestRId()));

        // 통계 퍼센티지 구하기
        List<TestResult> results = testRepository.findAll();
        Integer sum = 0;
        Integer target = 0;

        for (int i=0; i<results.size(); i++){
            sum+=results.get(i).getTestRCount();
            if(tr.getTestRId()==results.get(i).getTestRId()){
                target=results.get(i).getTestRCount();
            }
        }
        float percent = (target/(float)sum)*100;

        // JPA 의 영속성 컨텍스트 덕분에 entity 객체의 값만 변경하면 자동으로 변경사항 반영함!
        // 따라서 repository.update 를 쓰지 않아도 됨.
//        tr.plusCount();

        return TestResultDto.from(result, percent);
    }

    public float getResultPercent(int score){
        List<TestResult> results = testRepository.findAll();
        TestResultInterface tri = testRepository.selectResultContent(score);
        Integer sum = 0;
        Integer target = 0;

        for (int i=0; i<results.size(); i++){
            sum+=results.get(i).getTestRCount();
            if(tri.getTestRId()==results.get(i).getTestRId()){
                target=results.get(i).getTestRCount();
            }
        }

        float result = (target/(float)sum)*100;

        return result;
    }
}
