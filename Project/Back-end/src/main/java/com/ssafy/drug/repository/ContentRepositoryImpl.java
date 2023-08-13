package com.ssafy.drug.repository;

import com.querydsl.core.types.Projections;
import com.querydsl.jpa.impl.JPAQueryFactory;
import com.ssafy.drug.dto.ContentDetailDto;
import com.ssafy.drug.dto.ContentDto;
import com.ssafy.drug.dto.ContentTagDto;
import com.ssafy.drug.model.*;
import com.ssafy.drug.repository.querydsl.ContentRepositoryCustom;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;

import java.util.List;

import static com.querydsl.core.group.GroupBy.groupBy;
import static com.querydsl.core.group.GroupBy.list;

@RequiredArgsConstructor
public class ContentRepositoryImpl implements ContentRepositoryCustom {
    private final JPAQueryFactory queryFactory;

    @Override
    @Transactional
    public List<ContentDto> selectContents() {
        QContent c = QContent.content;
        QContentDetail cd = QContentDetail.contentDetail;
        QContentTag ct = QContentTag.contentTag;
        List<ContentDto> result = queryFactory
                .from(c)
                .join(cd).on(c.contentId.eq(cd.content.contentId)).orderBy(c.contentId.asc(),cd.contentOrder.asc())
                .transform(groupBy(c.contentId).list(Projections.constructor(ContentDto.class,
                        c.contentId, c.contentType, c.contentLike,
                        list(Projections.constructor(ContentDetailDto.class, cd.contentOrder, cd.contentImg)))));

        List<ContentTagDto> tags = queryFactory
                .from(c)
                .join(ct).on(c.contentId.eq(ct.content.contentId))
                .transform(groupBy(c.contentId).
                        list(Projections.constructor(ContentTagDto.class,
                                c.contentId,
                                list(Projections.constructor(String.class,ct.contentTagName)))));

        for(int i=0; i<result.size(); i++){
            if(result.get(i).getContentId()==tags.get(i).getContentId()) {
                result.get(i).setTags(tags.get(i).getContentName());
            }
        }
        return result;
    }
}
