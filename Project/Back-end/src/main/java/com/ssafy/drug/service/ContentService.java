package com.ssafy.drug.service;

import com.ssafy.drug.dto.ContentDto;
import com.ssafy.drug.repository.ContentRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@Transactional
public class ContentService {
    @Autowired
    private ContentRepository contentRepository;

    public List<ContentDto> selectContents(){
        return contentRepository.selectContents();
    }
}
