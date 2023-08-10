package com.ssafy.drug.service;

import com.ssafy.drug.repository.AiUrlRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
@Transactional
public class AiUrlService {
    @Autowired
    private AiUrlRepository aiUrlRepository;

    public String selectUrl(int id){
        return aiUrlRepository.findById(id).getUrl();
    }
}
