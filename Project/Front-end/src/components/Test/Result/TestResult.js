import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import { useLocation } from "react-router-dom";
import ai_img from "../../../assets/images/rec_ai.png";
import ar_img from "../../../assets/images/rec_ar.png";
import simul_img from "../../../assets/images/rec_.png";
import "./TestResult.css";

import Statistic from "components/common/Statistic/Statistic";
// import { BottomMenu } from "components/common/BottomMenu/BottomMenu";
import { PercentBar } from "components/common/Percentage/Percentage";
import { async } from "q";

export const AI = () => {
  const searchParams = new URLSearchParams(useLocation().search);
  const score = searchParams.get("score");
  const [img, setImg] = useState(null);
  const [title, setTitle] = useState("");
  const [subtitle, setSubtitle] = useState("");
  const [content, setContent] = useState("");
  const [percent, setPercent] = useState(0.0);

  const finalResult = async () => {
    // 현재 사용자의 번호 가져오기
    const user = await axios
      .get(process.env.REACT_APP_SERVER_API_URL + "/user/userno")
      .catch((Error) => {
        console.log(Error);
      });
    let userno = user.data + 1;

    // 사용자 점수 저장
    await axios({
      method: "POST",
      url: process.env.REACT_APP_SERVER_API_URL + "/user/result",
      headers: {
        "Content-Type": "application/json",
      },
      data: {
        user_id: userno,
        test_final_score: score,
      },
    })
      .then((response) => {
        console.log(response);
      })
      .catch((Error) => {
        console.log(Error);
      });

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
        <p className="text1">
          고려대학교 생명과학부 백자현 교수팀과 최세영 교수의 공동연구에 의해 스트레스는 마약 중독
          재발에 많은 영향을 끼친다는 사실이 밝혀졌습니다.
        </p>
      </div>
      <Statistic></Statistic>
      <div className="frame-div222">
        <div className="group-container222">
          <div className="group-parent222">
            <div className="rectangle-parent222">
              <div className="group-item222" />
              <Link to="/simul">
                <div className="group-inner222" style={{ backgroundImage: `url(${simul_img})` }} />
                <div className="ai222">SIMULATION</div>
              </Link>
            </div>
            <div
              className="rectangle-container222"
              // onClick={handleARClick}
            >
              <div className="group-item222" />
              <Link to="/ar/filter2">
                <div className="group-inner222" style={{ backgroundImage: `url(${ar_img})` }} />
                <div className="ai222">AR</div>
              </Link>
            </div>
            <div className="rectangle-group222">
              <div className="group-item222" />
              <Link to="/ai/upload">
                <div className="group-inner222" style={{ backgroundImage: `url(${ai_img})` }} />
                <div className="ai222">AI</div>
              </Link>
            </div>
            {/* <div className="rectangle-parent2221"> */}
            {/* <div className="group-item222" />
              <div className="group-inner222" />
              <div className="ai222">...</div>
            </div> */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AI;
