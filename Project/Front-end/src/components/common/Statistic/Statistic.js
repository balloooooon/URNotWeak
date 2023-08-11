import React, {useState, useEffect} from "react";
import axios from "axios";
import {useLocation, useNavigate} from "react-router-dom";
import './Statistic.css'
import useCountNum from './UseCountUp';
import { calculatePredictedDeaths } from './DrugDeathStatistics';

export const AI = () => {
  const [crimeData, setCrimeData] = useState([]);

  const searchParams = new URLSearchParams(useLocation().search);
  const score = searchParams.get('score');
  const navigate = useNavigate();
  const drugDeathCount = useCountNum(calculatePredictedDeaths().predictedDeathsToday);
  const drugCrimeCount = useCountNum(crimeData);

  

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

    axios.get(process.env.REACT_APP_DRUG_CRIME_URL +
      '/15088286/v1/uddi:4522eba6-11cc-4c80-89fa-8b47bf6d691f?page=1&perPage=10&serviceKey=' + 
      process.env.REACT_APP_DRUG_CRIME_KEY)
    .then(({data})=>{
      const year2015Data = data.data.find(item => item.년도 === 2022); // 22년도의 검거 건수를 가져옴
      //console.log(year2015Data["검거 건수"])
      setCrimeData([year2015Data["검거 건수"]]);
    })
    .catch((Error)=>{console.log(Error)});
  }, []);

  

  return (
    <div className="statistic mt">
      <div className="line"></div>
      <p className="text1">통계자료</p>
      <div className="flex">
        <div className="st-item">
          <p className="text-n">마약으로 인한 <br/>사망자수</p>
          <p className="text-m">{drugDeathCount}명</p>
        </div>
        <div className="st-item">
          <p className="text-n">마약 사건<br/>발생 건수</p>
          <p className="text-m">{drugCrimeCount}건</p>
        </div>
      </div>
    </div>
  );
};
    
export default AI;
