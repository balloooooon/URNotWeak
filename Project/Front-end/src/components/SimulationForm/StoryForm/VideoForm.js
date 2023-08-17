import React, { useState, useEffect } from "react";
import axios from "axios";
import SelectTxt from "./SelectTxt";
import MonoForm from "./MonoForm";
import HandForm from "./HandForm";
import AR1 from "components/AR filter/filter1";
import AR2 from "components/AR filter/filter2";
import AR3 from "components/AR filter/filter3";
import AR4 from "components/AR filter/filter4";
import AR5 from "components/AR filter/filter5";
import ResultForm from "../ResultForm/ResultForm";
import NextBtn from "../NextBtn";
import "./VideoForm.css";

const VideoForm = ({ selectedIndex }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [simulData, setSimulData] = useState([]);
  const [simulTxt, setSimulTxt] = useState([]);
  const [videoRef, setVideoRef] = useState(null);
  const [showResultForm, setShowResultForm] = useState(false);
  const arComponents = [AR1, AR2, AR3, AR4, AR5];

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
  }, [currentStep, selectedIndex]);

  const handleNextClick = () => {
    if (currentStep < simulData.length - 1) {
      setCurrentStep((prevStep) => prevStep + 1);
    } else {
      setShowResultForm(true);
    }
  };

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

      video.onended = handleNextClick;
    }
  }, [videoRef, currentStep, simulData]);

  // VideoRef나 currentStep이나 simulData가 바뀌면 비디오 정보도 바뀌게 설정

  let content = null;

  if (!showResultForm && simulData.length > 0) {
    const currentStepData = simulData[currentStep];

    if (currentStepData.simulContentType === 1) {
      content = (
        <div className="video-story">
          <NextBtn onClick={handleNextClick} />
          <div className="videoform-wrapper">
            <div className="videoform">
              <video
                ref={(ref) => setVideoRef(ref)}
                autoPlay
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
          </div>
          <div className="frame-child001">
            <div className="chapter1-txt tx-b">{`${currentStepData.simulText}. ${currentStepData.simulAnswer1}`}</div>
          </div>
          <div className="url-group">
            {currentStepData.sourceName && (
              <p className="video_url">{`${currentStepData.sourceName}`}</p>
            )}
            {currentStepData.sourceUrl && (
              <p className="video_url2">{`${currentStepData.sourceUrl}`}</p>
            )}
          </div>
        </div>
      );
    } else if (currentStepData.simulContentType === 2) {
      content = (
        <SelectTxt
          text={currentStepData.simulText}
          answer1={currentStepData.simulAnswer1}
          answer2={currentStepData.simulAnswer2}
          shift1={currentStepData.simulShift1}
          shift2={currentStepData.simulShift2}
          onNext={handleNextClick}
        />
      );
    } else if (currentStepData.simulContentType === 3) {
      console.log("mono", currentStepData.simulText);
      content = <MonoForm text={simulTxt} onNext={handleNextClick} />;
    } else if (currentStepData.simulContentType === 4) {
      console.log("ar으로 이동");
      const newRandomIndex = Math.floor(Math.random() * 1) + 5;
      let ARComponent;
      if (newRandomIndex === 1) {
        ARComponent = AR1;
      } else if (newRandomIndex === 2) {
        ARComponent = AR2;
      } else if (newRandomIndex === 3) {
        ARComponent = AR3;
      } else if (newRandomIndex === 4) {
        ARComponent = AR4;
      } else if (newRandomIndex === 5) {
        ARComponent = AR5;
      }
      content = React.createElement(ARComponent, { onNext: handleNextClick });
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
