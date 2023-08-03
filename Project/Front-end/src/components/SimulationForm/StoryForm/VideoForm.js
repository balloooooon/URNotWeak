import React, { useState, useEffect } from "react";
import axios from "axios";
import SelectTxt from "./SelectTxt";
import MonoForm from "./MonoForm";
import NextBtn from "../NextBtn";
import "./VideoForm.css";

const VideoForm = ({ selectedIndex }) => {
  const [simulData, setSimulData] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [videoRef, setVideoRef] = useState(null);

  useEffect(() => {
    const fetchSimulData = async () => {
      try {
        const response = await axios.get(
          `http://43.202.55.53:8589/simulation/${selectedIndex}`
        );
        setSimulData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchSimulData();
  }, [selectedIndex]);

  const handleNextClick = () => {
    if (currentStep < simulData.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  useEffect(() => {
    if (videoRef && simulData.length > 0) {
      const video = videoRef;
      const newSource = simulData[currentStep].simulVideoUrl;

      // 비디오를 중지한 후에 바꾸어야 합니다.
      video.pause();
      video.src = newSource;
      video.load();

      // 비디오가 준비되면 자동으로 재생합니다.
      video.onloadeddata = () => {
        video.play();
      };
    }
  }, [videoRef, currentStep, simulData]);

  return (
    <>
      {simulData.length > 0 && (
        <>
          {simulData[currentStep].simulContentType === 1 ? (
            <div className="video-story">
              <div className="videoform-wrapper">
                <div className="videoform">
                  <video
                    ref={(ref) => setVideoRef(ref)}
                    autoPlay
                    loop
                    style={{ objectFit: "cover", width: "100%", height: "100%" }}
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
                <div className="chapter1-txt">{`Chapter ${simulData[currentStep].simulOrder}. ${simulData[currentStep].simulText}`}</div>
              </div>
            </div>
          ) : (
            <>
              {simulData[currentStep].simulContentType === 2 && (
                <SelectTxt
                  text={simulData[currentStep].simulText}
                  answer1={simulData[currentStep].simulAnswer1}
                  answer2={simulData[currentStep].simulAnswer2}
                  onNext={handleNextClick}
                />
              )}
              {simulData[currentStep].simulContentType === 3 && (
                <MonoForm text={simulData[currentStep].simulText} />
              )}
            </>
          )}
        </>
      )}
    </>
  );
};

export default VideoForm;
