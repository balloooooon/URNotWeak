import { useState } from "react";
import video1 from "../../../assets/images/pregnant1.mp4";
import NextBtn from "../NextBtn";
import "./VideoForm.css";

const VideoForm = () => {
  return (
    <div className="video-story">
      <div className="videoform-wrapper">
        <div className="videoform">
          <video controls>
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
  );
};

export default VideoForm;
