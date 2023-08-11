package com.ssafy.drug.repository;

import com.ssafy.drug.model.AiUrl;
import org.springframework.data.jpa.repository.JpaRepository;

public interface AiUrlRepository extends JpaRepository<AiUrl, Integer> {
    public AiUrl findById(int id);
}
