package com.ssafy.drug.repository;

import com.ssafy.drug.model.Simulation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SimulResultRepository extends JpaRepository<Simulation, Integer> {
    public Simulation findBySimulNo(int simulNo);
}
