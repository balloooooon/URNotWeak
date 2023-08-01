import { useState } from "react";
import video1 from "../../../assets/images/pregnant1.mp4";
import NextBtn from "../NextBtn";
import SelectTxt from "./SelectTxt";
import "./VideoForm.css";

const VideoForm = () => {
  const [VideoEnded, setVideoEnded] = useState(false);
  const handleVideoEnded = () => {
    setVideoEnded(true);
  };
  return (
    <>
      {!VideoEnded ? (
        <div className="video-story">
          <div className="videoform-wrapper">
            <div className="videoform">
              <video controls onEnded={handleVideoEnded}>
                <source src={video1} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <NextBtn />
          </div>
          <div className="frame-child">
            <div className="chapter1-txt">Chapter1. 마약투여</div>
          </div>
        </div>
      ) : (
        <SelectTxt />
      )}
    </>
  );
};

export default VideoForm;
