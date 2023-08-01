import { useCallback } from "react";
import "./HandForm.css";
const HandForm = () => {
  const onM324menuIconClick = useCallback(() => {
    // Please sync "Android - 4" to the project
  }, []);

  return (
    <div className="android-82">
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
      <div className="httpslottiefilescomanimat-parent">
        <img
          className="httpslottiefilescomanimat-icon"
          alt=""
          src="/httpslottiefilescomanimationsshieldupuejjv2msnp@2x.png"
        />
        <div className="group-parent">
          <img className="group-child" alt="" src="/group-64.svg" />
          <img
            className="httpslottiefilescomanimat-icon1"
            alt=""
            src="/httpslottiefilescomanimationsshieldupuejjv2msnp1@2x.png"
          />
<<<<<<< HEAD
          <div className="div2">드래그하여 내려주세요</div>
=======
          <div className="div2">드래그하여 내려주세요 !</div>
>>>>>>> 0d39bd374d7f4fb84da185e15117ce221ecac3cc
        </div>
      </div>
    </div>
  );
};

export default HandForm;
