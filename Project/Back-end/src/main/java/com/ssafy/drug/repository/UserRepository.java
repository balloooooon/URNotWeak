package com.ssafy.drug.repository;

import com.ssafy.drug.model.User;
import com.ssafy.drug.repository.querydsl.UserRepositoryCustom;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<User, Integer>, UserRepositoryCustom {

    public int findById(int id);
}
