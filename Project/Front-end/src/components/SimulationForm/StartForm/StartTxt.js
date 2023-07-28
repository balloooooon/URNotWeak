import { useCallback } from "react";
import SendImg from "../../../assets/images/send.svg";
import "./StartTxt.css";
const StartTxt = () => {
  const onM324menuIconClick = useCallback(() => {
    // Please sync "Android - 4" to the project
  }, []);

  const onFrameContainer8Click = useCallback(() => {
    // Please sync "Android - 85" to the project
  }, []);

  return (
    <div className="android-87">
      <div className="status-bar">
        <div className="div">12:45</div>
        <div className="icons">
          <img className="m324wifi-icon" alt="" src="/m324wifi.svg" />
          <img
            className="m324wifi-icon"
            alt=""
            src="/m324signal-cellular.svg"
          />
          <img className="m324wifi-icon" alt="" src="/m324battery-full.svg" />
        </div>
      </div>
      <div className="status-bar">
        <div className="div">약해지지마</div>
        <img
          className="m324menu-icon"
          alt=""
          src="/m324menu.svg"
          onClick={onM324menuIconClick}
        />
      </div>
      <div className="frame-parent">
        <div className="frame-group">
          <div className="frame">
            <div className="div">야! 지금 클럽인데 빨리 나와!</div>
          </div>
          <div className="frame">
            <div className="div">돈 챙겨서</div>
          </div>
          <div className="frame">
            <div className="div">오늘 새로운거 나왔대</div>
          </div>
          <div className="frame">
            <div className="div">빨리와</div>
          </div>
        </div>
        <div className="frame-wrapper">
          <div className="wrapper">
            <div className="div">오</div>
          </div>
        </div>
      </div>
      <div className="group-parent" onClick={onFrameContainer8Click}>
        <div className="rectangle-parent">
          <div className="group-child" />
          <div className="div7">지금 출발한다</div>
        </div>
        <img className="icon-send" alt="" src={SendImg} />
      </div>
    </div>
  );
};

export default StartTxt;
