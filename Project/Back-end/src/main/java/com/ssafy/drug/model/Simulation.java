package com.ssafy.drug.model;

import jakarta.persistence.*;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

@Data
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="simulation")
public class Simulation {
    @Id
    @Column(name="simul_no")
    private int simulNo;

    @Column(name="news_url")
    private String newsUrl;

    @OneToMany(mappedBy = "simulation", cascade = CascadeType.ALL)
    private List<SimulationDetail> simulationDetails = new ArrayList<>();

    @Builder
    public Simulation(int simulNo, String newsUrl) {
        this.simulNo = simulNo;
        this.newsUrl = newsUrl;
    }
}
