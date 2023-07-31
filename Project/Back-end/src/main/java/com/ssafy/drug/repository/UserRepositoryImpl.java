package com.ssafy.drug.repository;

import com.querydsl.jpa.impl.JPAQueryFactory;
import com.ssafy.drug.dto.QUser;
import com.ssafy.drug.repository.querydsl.UserRepositoryCustom;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public class UserRepositoryImpl implements UserRepositoryCustom {
    private final JPAQueryFactory queryFactory;

    @Override
    public Long countUser() {
        QUser qUser = QUser.user;
        Long userNo = queryFactory
                .select(qUser.count())
                .from(qUser)
                .fetchFirst();
        return  userNo;
    }
}
