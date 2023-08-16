package com.ssafy.drug.model;

import jakarta.persistence.*;
import lombok.*;
import org.antlr.v4.runtime.misc.NotNull;

@Data
@Getter
@AllArgsConstructor
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="simulation_detail")
public class SimulationDetail {

    @Id
    @Column(name="simul_detail_id")
    private Integer simulDetailId;

    @ManyToOne(optional = false)
    @JoinColumn(name="simul_no")
    @NotNull
    Simulation simulation;

//    @Column(name="simul_no")
//    @NotNull
//    private int simulNo;

    @Column(name="simul_order")
    @NotNull
    private Integer simulOrder;

    @Column(name="simul_content_type")
    private Integer simulContentType;

    @Column(name="simul_video_url")
    private String simulVideoUrl;

    @Column(name="simul_thumbnail")
    private Integer simulThumbnail;

    @Column(name="simul_text")
    private String simulText;

    @Column(name="simul_answer1")
    private String simulAnswer1;

    @Column(name="simul_answer2")
    private String simulAnswer2;

    @Column(name="simul_shift1")
    private Integer simulShift1;

    @Column(name="simul_shift2")
    private Integer simulShift2;

    @Column(name="ar_type")
    private Integer arType;

    @Column(name="source_name")
    private String sourceName;

    @Column(name="source_url")
    private String sourceUrl;
}
