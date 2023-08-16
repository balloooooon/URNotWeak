import React, { useState, useCallback, useEffect } from "react";
import SendImg from "../../../assets/images/send.svg";
import VideoForm from "../StoryForm/VideoForm";
import "./StartTxt.css";

const StartTxt = ({ selectedIndex }) => {
  const [isContentVisible, setIsContentVisible] = useState(false);
  const [showVideoForm, setShowVideoForm] = useState(false);

  const onFrameContainer8Click = useCallback(() => {
    setIsContentVisible(!isContentVisible);
  }, [isContentVisible]);

  useEffect(() => {
    if (isContentVisible) {
      const timer = setTimeout(() => {
        setIsContentVisible(false);
        setShowVideoForm(true);
      }, 2000);

      return () => clearTimeout(timer);
    }
  }, [isContentVisible]);

  return (
    <div className="android-87">
      <div className={`frame-parent ${showVideoForm ? "hidden" : ""}`}>
        <div className="frame-group">
          <div className="frame">
            <div className="div tx-t">야! 지금 클럽인데 빨리 나와!</div>
          </div>
          <div className="frame">
            <div className="div tx-t">돈 챙겨서</div>
          </div>
          <div className="frame">
            <div className="div tx-t">오늘 새로운거 나왔대</div>
          </div>
          <div className="frame">
            <div className="div tx-t">빨리와</div>
          </div>
        </div>
        <div className={`frame-wrapper ${showVideoForm ? "hidden" : ""}`}>
          <div className="wrapper12">
            <div className="div1225 tx-t">오</div>
          </div>
          {isContentVisible && !showVideoForm && (
            <div className="wrapper13">
              <div className="div1225 tx-t">지금 출발한다</div>
            </div>
          )}
        </div>
      </div>
      <div
        className={`group-parent01 ${
          isContentVisible || showVideoForm ? "hidden" : ""
        }`}
        onClick={onFrameContainer8Click}
      >
        <div className={`rectangle-parent ${isContentVisible ? "hidden" : ""}`}>
          <div className={`group-child1 ${isContentVisible ? "hidden" : ""}`} />
          {!isContentVisible && !showVideoForm && (
            <div className={`div7 tx-t ${isContentVisible ? "hidden" : ""}`}>
              지금 출발한다
            </div>
          )}
          <img className="icon-send" alt="" src={SendImg} />
        </div>
      </div>

      {showVideoForm && <VideoForm selectedIndex={selectedIndex} />}
    </div>
  );
};

export default StartTxt;
