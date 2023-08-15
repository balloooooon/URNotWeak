import React, { useState, useEffect } from "react";
import CallBtn from "../../../assets/Group54.svg";
import RejectBtn from "../../../assets/images/reject_btn.png";
import CallingImg from "../../../assets/images/white_phone.png";
import RedBtn from "../../../assets/images/redBtn.png";
import VideoForm from "../StoryForm/VideoForm";
import voice from "../../../assets/audio/hy_voice.m4a";
import ringback from "../../../assets/audio/ringback.wav";
import "./StartAnd.css";

const StartAnd = ({ selectedIndex }) => {
  const [count, setCount] = useState(0);
  const [isCallBtn, setCallBtn] = useState(true);
  const [isRedBtn, setRedBtn] = useState(false);
  const [showVideoForm, setShowVideoForm] = useState(false);
  const [audio, setAudio] = useState(null);
  const [ringbackAudio, setRingbackAudio] = useState(null);
  let interval;

  const handleCallBtn = () => {
    if (ringbackAudio) {
      ringbackAudio.pause();
      ringbackAudio.currentTime = 0;
    }
    setCallBtn(false);
    setRedBtn(true);
    setCount(0);
    const audio = new Audio(voice);
    setAudio(audio);
    audio.play();
  };
  const handleRedBtn = () => {
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    setShowVideoForm(true);
    clearInterval(interval);
  };

  useEffect(() => {
    const ringbackAudio = new Audio(ringback);
    setRingbackAudio(ringbackAudio);
    ringbackAudio.loop = true;
    ringbackAudio.play();
  }, []);

  useEffect(() => {
    if (!isCallBtn) {
      interval = setInterval(() => {
        setCount((prevCount) => prevCount + 1);
      }, 1000);

      if (count >= 7) {
        setShowVideoForm(true);
        clearInterval(interval);
      }

      return () => clearInterval(interval);
    }
  }, [count, isCallBtn]);

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0");
    const remaining = (seconds % 60).toString().padStart(2, "0");
    return `${minutes}:${remaining}`;
  };

  return (
    <div className="start1">
      {showVideoForm ? null : (
        <div className="call-wrapper">
          {isCallBtn ? (
            <div className="calling-txt">통화 수신중</div>
          ) : (
            <div className="count-time">
              <img className="call-img" src={CallingImg} alt="Calling" />
              {formatTime(count)}
            </div>
          )}
          <div className="phone-num">02-1234-56**</div>
        </div>
      )}

      {showVideoForm ? null : (
        <>
          {isRedBtn ? (
            <img
              className="red-btn"
              alt=""
              src={RedBtn}
              onClick={handleRedBtn}
            />
          ) : (
            <>
              <img
                className="call-btn1"
                alt=""
                src={CallBtn}
                onClick={handleCallBtn}
              />
              <img className="call-btn2" alt="" src={RejectBtn} />
            </>
          )}
        </>
      )}

      {showVideoForm && <VideoForm selectedIndex={selectedIndex} />}
    </div>
  );
};

export default StartAnd;
