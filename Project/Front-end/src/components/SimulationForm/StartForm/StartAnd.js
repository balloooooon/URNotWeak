import React, { useState, useEffect } from "react";
import CallBtn from "../../../assets/Group54.svg";
import RejectBtn from "../../../assets/images/reject_btn.png";
import CallingImg from "../../../assets/images/white_phone.png";
import RedBtn from "../../../assets/images/redBtn.png";
import VideoForm from "../StoryForm/VideoForm";
import Nav from "components/Nav/Nav";
import "./StartAnd.css";

const StartAnd = () => {
  const [count, setCount] = useState(0);
  const [isCallBtn, setCallBtn] = useState(true);
  const [isRedBtn, setRedBtn] = useState(false);
  useEffect(() => {
    const interval = setInterval(() => {
      setCount((prevCount) => prevCount + 1);
    }, 1000);
    if (count == 3) {
      clearInterval(interval);
      setCallBtn(false);
      setRedBtn(true);
    }

    return () => clearInterval(interval);
  }, []);
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0");
    const remaining = (seconds % 60).toString().padStart(2, "0");
    return `${minutes}:${remaining}`;
  };

  const handleCallBtn = () => {
    setCallBtn(false);
    setRedBtn(true);
  };

  return (
    <div className="start1">
      <Nav />
      <div className="call-wrapper">
        {isCallBtn ? (
          <div className="calling-txt">통화 수신중</div>
        ) : (
          <div className="count-time">
            <img className="call-img" src={CallingImg} />
            {formatTime(count)}
          </div>
        )}
        <div className="phone-num">02-1234-56**</div>
      </div>

      {isRedBtn ? (
        <img className="red-btn" alt="" src={RedBtn} />
      ) : (
        <div className="group-parent1">
          <img
            className="call-btn1"
            alt=""
            src={CallBtn}
            onClick={handleCallBtn}
          />
          <img className="call-btn2" alt="" src={RejectBtn} />
        </div>
      )}

      {count >= 3 && <VideoForm />}
    </div>
  );
};

export default StartAnd;
