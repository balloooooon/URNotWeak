import React, {useState, useEffect} from "react";
import axios from "axios";
import {useLocation} from "react-router-dom";
import './TestResult.css'

import Statistic from 'components/common/Statistic/Statistic';
import { BottomMenu } from "components/common/BottomMenu/BottomMenu";
import { PercentBar } from 'components/common/Percentage/Percentage';

export const AI = () => {
  const searchParams = new URLSearchParams(useLocation().search);
  const score = searchParams.get('score');
  const [img, setImg] = useState(null);
  const [title, setTitle] = useState('');
  const [subtitle, setSubtitle] = useState('');
  const [content, setContent] = useState('');
  const [percent, setPercent] = useState(0.0);

  useEffect(() => {
    axios.get(process.env.REACT_APP_SERVER_API_URL+'/test/result?score='+score)
    .then(({data})=>{
      console.log(data);
      // question = data;
      // setQuestion(Object.values(data)); //Object.entries
      setImg(data.testRImg);
      setTitle(data.testRTitle);
      setSubtitle(data.testRSubtitle);
      setContent(data.testRContent);
      setPercent(data.percent.toFixed(2));  //소수점 버림
    })
    .catch((Error)=>{console.log(Error)});
  }, []);

  return (
    <div className="result-page">
      <img className="result" src={img} />
      <div className="text-content">
        <p className="text1 title">{title}</p>
        <p className="text1">{subtitle}</p>
        <p className="text1">{content}</p>
      </div>
      
      <div className="percent mt">
        <PercentBar bgcolor="#ef6c00" completed={percent}></PercentBar>
        <p className="text-s">{percent}의 사용자가 같은 결과를 가지고 있습니다.</p>
      </div>

      <div className="text-content mt">
        <div className="line"></div>
        <p className="text1">마약과 스트레스</p>
        <p className="text1">고려대학교 생명과학부 백자현 교수팀과 
최세영 교수의 공동연구에 의해 스트레스는
마약 중독 재발에 많은 영향을
끼친다는 사실이 밝혀졌습니다.</p>
      </div>

      <Statistic></Statistic>
      <BottomMenu />
    </div>
  );
};
    
export default AI;
