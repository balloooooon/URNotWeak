import React, { useState, useEffect } from "react";
import "react-toastify/dist/ReactToastify.css";
import './List.css'

export const ContentList = () => {
    //리스트 가져오기 만들어야함
    const list = [
        {
            id:1,
            img:"https://img.freepik.com/free-photo/cute-little-dog-isolated-on-yellow_23-2148985931.jpg",
            like:10,//좋아요수
            view:100, //조회수
            tag:["#약물정보","#펜타닐"] //태그들
        },
        {
            id:2,
            img:"https://image.utoimage.com/preview/cp872722/2022/12/202212008462_500.jpg",
            like:20,//좋아요수
            view:200, //조회수
            tag:["#약물정보","#펜타닐"] //태그들
        },
        {
            id:3,
            img:"https://img.freepik.com/free-photo/front-view-adorable-shiba-inu-dog_23-2149457807.jpg?w=360&t=st=1691718094~exp=1691718694~hmac=06fdac5e70ad01f4bac6b23f4d5de158656fd9d0c99fbea5f77cc18ddec9dfd7",
            like:30,//좋아요수
            view:300, //조회수
            tag:["#약물정보","#펜타닐"] //태그들
        }
    ]

  return (
    <div className="contents">
      <div className="tx-b-b title">컨텐츠</div>
      <div className="items">
        {list.map(item => {
            return (
                <div className="item" key={item.id}>
                    <figure className="image-wrapper">
                        <img src={item.img}></img>
                    </figure>
                    <div>{item.like}</div>
                    <div>{item.view}</div>
                    
                    <div className="tags">
                    {item.tag.map(t => {
                        return(
                            <div>{t}</div>
                        )
                    })}
                    </div>

                </div>
            )
        }
        )}
      </div>
    </div>
  );
};

export default ContentList;
