import { useCallback } from "react";
import "./StartAnd.css";

const StartAnd = () => {
  const onM324menuIconClick = useCallback(() => {
    // Please sync "Android - 4" to the project
  }, []);

  return (
    <div className="android-92">
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
      <div className="group-parent">
        <div className="wrapper">
          <div className="div2">02-1234-56**</div>
        </div>
        <img className="frame-child" alt="" src="./assets/Group54.svg" />
        <img className="frame-item" alt="" src="/vector-15.svg" />
        <div className="div3">통화 수신중</div>
      </div>
    </div>
  );
};

export default StartAnd;
