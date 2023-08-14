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
    }

  return (
    <div className="contents">
      <div className="tx-b-b title">컨텐츠</div>
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
                        <p className="tx-s-b">{item.contentLike}</p>
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
  );
};

export default ContentList;
