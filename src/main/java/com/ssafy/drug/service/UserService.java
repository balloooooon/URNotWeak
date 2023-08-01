package com.ssafy.drug.service;

import com.ssafy.drug.model.User;
import com.ssafy.drug.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class UserService {
    @Autowired
    private UserRepository testRepository;

    public List<User> findAll(){
        List<User> users = new ArrayList<>();
        testRepository.findAll().forEach(e -> users.add(e));
        return users;
    }

    public Long countUser(){
        return testRepository.countUser();
    }
}
