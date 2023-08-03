import { useState, useEffect } from "react";
import axios from "axios";
import SelectTxt from "./SelectTxt";
import MonoForm from "./MonoForm";
import NextBtn from "../NextBtn";
import "./VideoForm.css";

const VideoForm = ({ selectedIndex }) => {
  const [simulData, setSimulData] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);

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

  // 확인
  const handleNextClick = () => {
    if (currentStep < simulData.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  return (
    <>
      {simulData.length > 0 && (
        <>
          {simulData[currentStep].simulContentType === 1 ? (
            <div className="video-story">
              <div className="videoform-wrapper">
                <div className="videoform">
                  <video
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
