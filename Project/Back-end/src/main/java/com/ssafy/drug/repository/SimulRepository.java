package com.ssafy.drug.repository;

import com.ssafy.drug.model.SimulationDetail;
import com.ssafy.drug.model.TestResult;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface SimulRepository extends JpaRepository<SimulationDetail, Integer> {
    public List<SimulationDetail> findBySimulThumbnail(int no);
}
