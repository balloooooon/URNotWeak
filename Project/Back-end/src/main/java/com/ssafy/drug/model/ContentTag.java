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
@Entity(name="content_tag")
public class ContentTag {
    @Id
    @Column(name="content_tag_id")
    private int contentTagId;

    @ManyToOne(optional = false)
    @JoinColumn(name = "content_id")
    @NotNull
    private Content content;

    @Column(name="content_tag_name")
    private String contentTagName;
}
