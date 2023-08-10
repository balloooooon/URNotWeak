// /* global Kakao */
import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import SelectForm from "../SelectForm/SelecctForm";
import axios from "axios";
import line from "../../../assets/images/line.png";
import twitter from "../../../assets/images/트위터.png";
import kakaoimg from "../../../assets/images/kakao.png";
import link from "../../../assets/images/share.png";
import ai_img from "../../../assets/images/rec_ai.png";
import ar_img from "../../../assets/images/rec_ar.png";
import test_img from "../../../assets/images/rec_.png";
// import news_3 from "../../../assets/images/news_club.jpg";
// import news_2 from "../../../assets/images/news_pregnant.jpg";
import news1 from "../../../assets/images/news_teen.JPG";

import "./ResultForm.css";

const ResultForm = ({ selectedIndex }) => {
  const [resultData, setResultData] = useState(null);
  const [randomImagePair, setRandomImagePair] = useState(null);
  const [showSelectForm, setShowSelectForm] = useState(false);
  const getRandomArFilterPage = () => {
    const randomIndex = Math.floor(Math.random() * 5) + 1;
    return `/ar/filter${randomIndex}`;
  };
  const [isNewsImgVisible, setIsNewsImgVisible] = useState(true);
  const handleNewsClick = () => {
    console.log("뉴스클릭확인");
    setIsNewsImgVisible(!isNewsImgVisible);
  };

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

  // useEffect(() => {
  //   Kakao.init(process.env.REACT_APP_KAKAO_API_KEY); // Kakao API 키 초기화
  // }, []);

  const copyToClipboard = (text) => {
    const el = document.createElement("textarea");
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand("copy");
    document.body.removeChild(el);
    window.alert("링크가 클립보드에 복사되었습니다.");
  };

  const shareToTwitter = (url) => {
    const encodedUrl = encodeURIComponent(url);
    console.log("Encoded URL:", encodedUrl);
    // const twitterUrl = `https://twitter.com/intent/tweet?url=${encodedUrl}`;
    const twitterUrl = `https://twitter.com/intent/tweet?url=https://urnotweak.site/ 약해지지마 시뮬레이션 체험해보기 !`;
    window.open(twitterUrl, "_blank");
  };

  const urlToShare = "https://urnotweak.site/";

  const handleKakaoShare = () => {
    if (window.Kakao) {
      window.Kakao.Link.sendDefault({
        objectType: "feed",
        content: {
          title: "친구에게 공유하기",
          description: "약해지지마 - 시뮬레이션을 체험하시겠습니까?",
          imageUrl: kakaoimg,
          link: {
            mobileWebUrl: "https://urnotweak.site",
            webUrl: "https://urnotweak.site",
          },
        },
        buttons: [
          {
            title: "웹으로 보기",
            link: {
              mobileWebUrl: "https://urnotweak.site",
              webUrl: "https://urnotweak.site",
            },
          },
        ],
      });
    }
  };

  if (!resultData || !randomImagePair) {
    return null;
  }

  if (showSelectForm) {
    return <SelectForm />;
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
            {isNewsImgVisible ? (
              <img
                className="result-img"
                src={news1}
                alt="News 1"
                onClick={handleNewsClick}
              />
            ) : (
              <video
                className="result-video"
                autoPlay
                onClick={handleNewsClick}
              >
                <source src={resultData.news} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            )}
          </div>
            <p className="api222">{`이 시뮬레이션은 허구가 아닌 `}</p>
            <p className="api222">사실기반의 체험입니다.</p>
          <div className="frame-child222" />
          {/* <img className="frame-item222" alt="" src={line} /> */}
        </div>
        <div className="container222">
          <div className="div2223">마약을하면..</div>
          <img className="frame-inner222" alt="" src={line} />
          <img className="icon222" alt="" src={randomImagePair.drugBeforeImg} />
          <img className="icon223" alt="" src={randomImagePair.drugAfterImg} />
          <div className="buttonslight-parent222">
            <div className="buttonslight222" style={{ pointerEvents: "none" }}>
              <div
                className="button-wrapper2221"
                style={{ pointerEvents: "none" }}
              >
                <div className="div2221" style={{ pointerEvents: "none" }}>
                  전, 후 사진 AI 체험하기
                </div>
              </div>
            </div>
            <div className="buttonslight2221">
              <div className="button-wrapper222">
                <div className="div222" onClick={() => setShowSelectForm(true)}>
                  다른 스토리 확인하기
                </div>
              </div>
            </div>
          </div>
          <img className="vector-icon222" alt="" src={line} />
        </div>
        <div className="group-wrapper222">
          <div className="group-div222">
            <div className="div2224">친구에게 공유하기</div>
            <img
              className="kakaotalk222"
              alt="카톡공유이미지"
              src={kakaoimg}
              onClick={handleKakaoShare}
            />
            <img
              className="twitter222"
              alt="트위터공유"
              src={twitter}
              onClick={() => shareToTwitter(urlToShare)}
            />
            <div className="group-child222" />
            <img
              className="share-url222"
              alt="링크공유"
              src={link}
              onClick={() => copyToClipboard(urlToShare)}
            />
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
              <div
                className="rectangle-parent222"
                // onClick={handleARClick}
              >
                <div className="group-item222" />
                <Link to={getRandomArFilterPage()}>
                  <div
                    className="group-inner222"
                    style={{ backgroundImage: `url(${ar_img})` }}
                  />
                  <div className="ai222">AR</div>
                </Link>
              </div>
              <div className="rectangle-group222">
                <div className="group-item222" />
                <Link to="/ai/upload">
                  <div
                    className="group-inner222"
                    style={{ backgroundImage: `url(${ai_img})` }}
                  />
                  <div className="ai222">AI</div>
                </Link>
              </div>
              <div className="rectangle-container222">
                <div className="group-item222" />
                <Link to="/test">
                  <div
                    className="group-inner222"
                    style={{ backgroundImage: `url(${test_img})` }}
                  />
                  <div className="ai222">TEST</div>
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
    </div>
  );
};

export default ResultForm;
