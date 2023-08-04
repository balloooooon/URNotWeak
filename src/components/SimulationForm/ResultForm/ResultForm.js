import React, { useState, useEffect } from "react";
import axios from "axios";
import line from "../../../assets/images/line.png";
import instagram from "../../../assets/images/instagram.png";
import kakao from "../../../assets/images/kakao.png";
import link from "../../../assets/images/share.png";
import "./ResultForm.css";

const ResultForm = ({ selectedIndex }) => {
  const [resultData, setResultData] = useState(null);
  const [randomImagePair, setRandomImagePair] = useState(null);


  useEffect(() => {
    const fetchResultData = async () => {
      try {
        const response = await axios.get(
          `https://www.urnotweak.site:8589/simulation/result/${selectedIndex}`
        );
        setResultData(response.data);
        const imagePair = getRandomImagePair(response.data);
        setRandomImagePair(imagePair);
      } catch (error) {
        console.error("Error fetching result data:", error);
      }
    };

    fetchResultData();
  }, [selectedIndex]);

  const getRandomImagePair = (data) => {
    const images = data.drugBaImgs;
    const randomIndex = Math.floor(Math.random() * images.length);
    const imagePair = images[randomIndex];
    return imagePair;
  };

  if (!resultData || !randomImagePair) {
    return null;
  }
  return (
    <div className="result222">
      <div className="frame-parent222">
        <div className="api-wrapper222">
          <div className="api-container222">
            <p className="api222">
              매년 (api)명의 중독자와 사망자가 발생하고 있습니다.
            </p>
            <p className="api222">&nbsp;</p>
            <p className="api222">당신은 안전할까요?</p>
          </div>
        </div>
        <div className="group222">
          <div className="div2222">
            <video className="result-video" autoPlay loop controls>
              <source src={resultData.news} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
            <p className="api222">{`이 시뮬레이션은 허구가 아닌 `}</p>
            <p className="api222">사실기반의 체험입니다.</p>
          </div>
          <div className="frame-child222" />
          {/* <img className="frame-item222" alt="" src={line} /> */}
        </div>
        <div className="container222">
          <div className="div2223">마약을하면..</div>
          <img className="frame-inner222" alt="" src={line} />
          <img className="icon222" alt="" src={randomImagePair.drugBeforeImg} />
          <img className="icon223" alt="" src={randomImagePair.drugAfterImg} />
          <div className="buttonslight-parent222">
            <div className="buttonslight222">
              <div className="button-wrapper222">
                <div className="div222">전, 후 사진 AI 체험하기</div>
              </div>
            </div>
            <div className="buttonslight2221">
              <div className="button-wrapper222">
                <div className="div222">다른 스토리 확인하기</div>
              </div>
            </div>
          </div>
          <img className="vector-icon222" alt="" src={line} />
        </div>
        <div className="group-wrapper222">
          <div className="group-div222">
            <div className="div2224">친구에게 공유하기</div>
            <img className="kakaotalk222" alt="카톡공유이미지" src={kakao} />
            <img className="instagram222" alt="인스타공유" src={instagram} />
            <div className="group-child222" />
            <img className="share-url222" alt="링크공유" src={link} />
          </div>
        </div>
        <div className="frame-div222">
          <div className="div2225">추천</div>
          <div
            className="buttonslight2222"
            // onClick={onButtonsLightContainer2Click}
          >
            <div className="button-wrapper222">
              <div className="div222">메인 페이지로</div>
            </div>
          </div>
          <div className="group-container222">
            <div className="group-parent222">
              <div className="rectangle-parent222">
                <div className="group-item222" />
                <div className="group-inner222" />
                <div className="ai">시뮬레이션</div>
              </div>
              <div className="rectangle-group222">
                <div className="group-item222" />
                <div className="group-inner222" />
                <div className="ai222">AI</div>
              </div>
              <div className="rectangle-container222">
                <div className="group-item222" />
                <div className="group-inner222" />
                <div className="ai222">채팅</div>
              </div>
              <div className="rectangle-parent2221">
                <div className="group-item222" />
                <div className="group-inner222" />
                <div className="ai222">...</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultForm;
