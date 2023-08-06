import React, { useState, useEffect } from "react";
import axios from "axios";
import SelectTxt from "./SelectTxt";
import MonoForm from "./MonoForm";
import HandForm from "./HandForm";
import AR from "../AR";
import ResultForm from "../ResultForm/ResultForm";
import NextBtn from "../NextBtn";
import "./VideoForm.css";

const VideoForm = ({ selectedIndex }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [simulData, setSimulData] = useState([]);
  const [simulTxt, setSimulTxt] = useState([]);
  const [videoRef, setVideoRef] = useState(null);
  const [showResultForm, setShowResultForm] = useState(false);

  useEffect(() => {
    const fetchSimulData = async () => {
      try {
        const response = await axios.get(
          `https://www.urnotweak.site:8589/simulation/${selectedIndex}`
        );
        setSimulData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchSimulData();
  }, [selectedIndex]);
//  selectedIndex가 변경될 때마다 simulData를 새로 가져옴


useEffect(() => {
  const fetchSimulTxt = async () => {
    try {
      const response = await axios.get(
        `https://www.urnotweak.site:8589/simulation/${selectedIndex}`
      );
      setSimulTxt(response.data[currentStep].simulText);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  fetchSimulTxt();
}, [currentStep,selectedIndex]);
  const handleNextClick = () => {
    if (currentStep < simulData.length - 1) {
      setCurrentStep(prevStep => prevStep + 1);
    } else {
      setShowResultForm(true);
    }
  };

// 안되네 ㅋㅋ 



  useEffect(() => {
    if (videoRef && simulData.length > 0) {
      const video = videoRef;
      const newSource = simulData[currentStep].simulVideoUrl;

      video.pause();
      video.src = newSource;
      video.load();

      video.onloadeddata = () => {
        video.play();
      };
    }
  }, [videoRef, currentStep, simulData]);

// VideoRef나 currentStep이나 simulData가 바뀌면 비디오 정보도 바뀌게 설정


  let content = null;

  if (!showResultForm && simulData.length > 0) {
    const currentStepData = simulData[currentStep];

    if (currentStepData.simulContentType === 1) {
      content = (
        <div className="video-story">
          <div className="videoform-wrapper">
            <div className="videoform">
              <video
                ref={(ref) => setVideoRef(ref)}
                autoPlay
                loop
                style={{
                  objectFit: "cover",
                  width: "100%",
                  height: "100%",
                }}
              >
                <source
                  src={simulData[currentStep].simulVideoUrl}
                  type="video/mp4"
                />
                Your browser does not support the video tag.
              </video>
            </div>
            <NextBtn onClick={handleNextClick} />
          </div>
          <div className="frame-child001">
            <div className="chapter1-txt">{`${currentStepData.simulText}. ${currentStepData.simulAnswer1}`}</div>
          </div>
        </div>
      );
    } else if (currentStepData.simulContentType === 2) {
      content = (
        <SelectTxt
          text={currentStepData.simulText}
          answer1={currentStepData.simulAnswer1}
          answer2={currentStepData.simulAnswer2}
          onNext={handleNextClick}
        />
      );
    } else if (currentStepData.simulContentType === 3) {
      console.log("mono",currentStepData.simulText)
      content = (
        <MonoForm text={simulTxt} onNext={handleNextClick} />
      );
    } else if (currentStepData.simulContentType === 4) {
      console.log("ar으로 이동")
      content = <AR onNext={handleNextClick} />;
    } else if (currentStepData.simulContentType === 5) {
      content = <HandForm onNext={handleNextClick} currentStep={currentStep} />;
    }
  }

  return (
    <>
      {content}
      {showResultForm && <ResultForm selectedIndex={selectedIndex} />}
    </>
  );
};

export default VideoForm;
