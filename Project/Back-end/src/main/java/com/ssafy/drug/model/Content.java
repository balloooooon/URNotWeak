package com.ssafy.drug.model;

import jakarta.persistence.*;
import lombok.*;

import java.util.ArrayList;
import java.util.List;

@Data
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Entity(name="content")
public class Content {
    @Id
    @Column(name="content_id")
    private int contentId;

    @Column(name="content_type")
    private String contentType;

    @Column(name="content_like")
    private int contentLike;

    @OneToMany(mappedBy = "content", cascade = CascadeType.ALL)
    private List<ContentDetail> contentDetails = new ArrayList<>();

    @Builder
    public Content(int contentId, String contentType, int contentLike) {
        this.contentId = contentId;
        this.contentType = contentType;
        this.contentLike = contentLike;
    }
}
