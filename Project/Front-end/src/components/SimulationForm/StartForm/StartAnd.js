import React, { useState, useEffect } from "react";
import CallBtn from "../../../assets/Group54.svg";
import RejectBtn from "../../../assets/images/reject_btn.png";
import CallingImg from "../../../assets/images/white_phone.png";
import RedBtn from "../../../assets/images/redBtn.png";
import "./StartAnd.css";

const StartAnd = () => {
  const [count, setCount] = useState(0);
  const [isCallBtn, setCallBtn] = useState(true);
  useEffect(() => {
    const interval = setInterval(() => {
      setCount((prevCount) => prevCount + 1);
    }, 1000);
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
  };

  return (
    <div className="start1">
      <div className="group-parent">
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
        <img
          className="call-btn1"
          alt=""
          src={CallBtn}
          onClick={handleCallBtn}
        />
        <img className="call-btn2" alt="" src={RejectBtn} />
        {/* <img className="frame-item" alt="" src="/vector-15.svg" /> */}
      </div>
    </div>
  );
};

export default StartAnd;
