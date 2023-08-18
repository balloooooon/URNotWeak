import React, { useState, useEffect } from "react";
import axios from "axios";
import { useLocation } from "react-router-dom";
import "./TestResult.css";

import Statistic from "components/common/Statistic/Statistic";
import { PercentBar } from "components/common/Percentage/Percentage";
import { BottomMenu } from "components/common/BottomMenu/BottomMenu";
import { Share } from 'components/common/Share/Share'

export const AI = () => {
  const searchParams = new URLSearchParams(useLocation().search);
  const score = searchParams.get("score");
  const [img, setImg] = useState(null);
  const [title, setTitle] = useState("");
  const [subtitle, setSubtitle] = useState("");
  const [content, setContent] = useState("");
  const [percent, setPercent] = useState(0.0);

  const finalResult = async () => {

    await axios
      .get(process.env.REACT_APP_SERVER_API_URL + "/test/result?score=" + score)
      .then(({ data }) => {
        console.log(data);
        // question = data;
        // setQuestion(Object.values(data)); //Object.entries
        setImg(data.testRImg);
        setTitle(data.testRTitle);
        setSubtitle(data.testRSubtitle);
        setContent(data.testRContent);
        setPercent(data.percent.toFixed(2)); //소수점 버림
      })
      .catch((Error) => {
        console.log(Error);
      });
  };

  useEffect(() => {
    finalResult();
  }, []);

  return (
    <div className="test-result-page">
      <p className="main tx-b">당신은...</p>
      <img className="result" src={img} />
      <div className="text-content">
        <p className="tx-b title">{title}</p>
        <p className="tx-r">{subtitle}</p>
        <p className="tx-t">{content}</p>
      </div>

      <div className="percent mt">
        <PercentBar bgcolor="#ef6c00" completed={percent}></PercentBar>
        <p className="tx-s">
          {percent}%의 사용자가 같은 결과를 가지고 있습니다.
        </p>
      </div>

      <div className="text-content mt">
        <div className="line"></div>
        <p className="tx-s">마약과 스트레스</p>
        <p className="tx-t">
          스트레스는 마약 중독 재발에 많은 영향을 끼친다는 연구 결과가 있습니다. 
          스트레스에 대한 대처 방법을 찾아 마약에 의존하지 않도록 주의하세요.
        </p>
      </div>
      <Statistic></Statistic>
      <Share></Share>
      <BottomMenu></BottomMenu>
    </div>
  );
};

export default AI;
