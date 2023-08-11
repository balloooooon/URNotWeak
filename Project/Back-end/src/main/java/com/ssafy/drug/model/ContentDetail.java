package com.ssafy.drug.model;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.antlr.v4.runtime.misc.NotNull;

@Data
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="content_detail")
public class ContentDetail {
    @Id
    @Column(name = "content_detail_id")
    private int contentDetailId;

    @ManyToOne(optional = false)
    @JoinColumn(name = "content_id")
    @NotNull
    private Content content;

    @Column(name = "content_order")
    private int contentOrder;

    @Column(name = "content_img")
    private String contentImg;
}
