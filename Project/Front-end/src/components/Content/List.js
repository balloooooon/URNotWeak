import React, { useState, useEffect } from "react";
import {ImageSlider} from './ImageSlider'
import axios from "axios";
import "react-toastify/dist/ReactToastify.css";
import './List.css'


import thumb0 from 'assets/images/icon-thumb0.svg'
import thumb1 from 'assets/images/icon-thumb1.svg'

export const ContentList = () => {
    const [list, setList] = useState([]);
    const [loading, setLoading] = useState(false);
    const [local, setLocal] = useState();
    const hashtag = ["약물정보", "중독", "코카인", "필로폰"];

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(process.env.REACT_APP_SERVER_API_URL + "/content")
                setList(response.data);
            } catch (error) {
                console.error("Error fetching result data:", error);
            }
        };

        // 초기 세팅 동안 loading표시되도록
        setLoading(true);

        // 데이터 가져오기
        fetchData();
        // 좋아요 표시한 내용 localStorage에서 가져오기
        var arr = localStorage.getItem('content-liked');
        if(arr == null || arr == undefined) arr = [];
        else arr = JSON.parse(arr);
        setLocal(arr);
        console.log(local)

        setLoading(false);

    }, [])

    const clickThumb = (id) => {
        var arr = local;
        arr.push(id);
        arr = [...arr];
        localStorage.setItem('content-liked', JSON.stringify(arr));
        setLocal(arr);

        // 좋아요 누르는 axios 코드 필요
        const response = axios.post(process.env.REACT_APP_SERVER_API_URL + `/content/like/${id}`);

    }


    const searchTag = (tag) => {
        
        setLoading(true);

        // 데이터 가져오기
        axios.get(process.env.REACT_APP_SERVER_API_URL + "/content/search", {params:{tag: tag}})
            .then((response) => {
                console.log(response.data);
                setList(response.data);
            })
            .catch((Error) => {console.log(Error)})
        

        setLoading(false);
    }

  return (
    <div className="contents">
        {loading?
        <div className="skeleton-container">
            <div className="tag-form skeleton"></div>
            <div className="simg skeleton"></div>
            <div className="slike skeleton"></div>
            <div className="shash skeleton"></div>
        </div>
        :
        <div>
        <div className="search-bar">
            {hashtag.map(tag => {
                return(
                    <div className="wrapper" key={tag} onClick={()=>searchTag(tag)}><a className="tx-t-b">#{tag}</a></div>
                )
            })}
        </div>
        
        <div className="scroll">
      <div className="items">
        {list.map(item => {
            return (
                <div className="item" key={item.contentId}>
                    <ImageSlider images={item.details}></ImageSlider>
                    
                    <div className="like">
                        {local.includes(item.contentId)?
                        <img src={thumb1}></img>:
                        <div onClick={()=>clickThumb(item.contentId)}><img src={thumb0}></img></div>
                        }
                        <p className="tx-s-b num">{item.contentLike}</p>
                    </div>

                    <div className="tags">
                    {item.tags.map(t => {
                        return(
                            <a>#{t} </a>
                        )
                    })}
                    </div>

                </div>
            )
        }
        )}
      </div>
         
      </div>
      </div>
        }
    </div>
  );
};

export default ContentList;
