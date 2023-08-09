import React, {useState, useEffect} from "react";
import axios from "axios";
import {useLocation, useNavigate} from "react-router-dom";
import './Statistic.css'
import useCountNum from './UseCountUp';
import { calculatePredictedDeaths } from './DrugDeathStatistics';

export const AI = () => {
  const searchParams = new URLSearchParams(useLocation().search);
  const score = searchParams.get('score');
  const navigate = useNavigate();
  const count = useCountNum(calculatePredictedDeaths().predictedDeathsToday);

  useEffect(() => {
    // axios.get(process.env.REACT_APP_SERVER_API_URL+'/test/result?score='+score)
    // .then(({data})=>{
    //   console.log(data);
    //   // question = data;
    //   // setQuestion(Object.values(data)); //Object.entries
    //   setImg(data.testRImg);
    //   setTitle(data.testRTitle);
    //   setSubtitle(data.testRSubtitle);
    //   setContent(data.testRContent);
    //   setPercent(data.percent);
    // })
    // .catch((Error)=>{console.log(Error)});
  }, []);

  

  return (
    <div className="statistic mt">
      <div className="line"></div>
      <p className="text1">통계자료</p>
      <div className="flex">
        <div className="st-item">
          <p className="text-n">마약으로 인한 <br/>사망자수</p>
          <p className="text-m">{count}명</p>
        </div>
        <div className="st-item">
          <p className="text-n">마약 사건<br/>발생 건수</p>
          <p className="text-m">2건</p>
        </div>
      </div>
    </div>
  );
};
    
export default AI;
