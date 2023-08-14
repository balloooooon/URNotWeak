package com.ssafy.drug.service;

import com.ssafy.drug.dto.ContentDetailDto;
import com.ssafy.drug.dto.ContentDto;
import com.ssafy.drug.dto.SimulThumbnailDto;
import com.ssafy.drug.model.Content;
import com.ssafy.drug.model.ContentDetail;
import com.ssafy.drug.repository.ContentDetailRepository;
import com.ssafy.drug.repository.ContentRepository;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@Service
@Transactional
public class ContentService {
    @Autowired
    private ContentRepository contentRepository;
    @Autowired
    private ContentDetailRepository contentDetailRepository;

    public List<ContentDto> selectContents(){
        return contentRepository.selectContents();
    }

    public void updateLike(int contentId) {
        Content content = contentRepository.findByContentId(contentId);
        content.plusCount();
    }

    public List<ContentDto> searchTag(String tag) {
        return contentRepository.searchTag(tag);
    }

    public String getImg() {
        List<ContentDetail> contents = contentDetailRepository.findAll();
        Random random = new Random();
        int num = random.nextInt(contents.size());
        return contents.get(num).getContentImg();
    }
}
