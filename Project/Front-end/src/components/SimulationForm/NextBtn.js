import { useCallback } from "react";
import NxtBtnImg from "../../assets/group-45.svg";
import "./NextBtn.css";
const NextBtn = () => {
  const onGroupIconClick = useCallback(() => {
    // Please sync "story3 -3" to the project
  }, []);

  return (
    <div className="android-88">
      <img
        className="android-88-child"
        alt=""
        src={NxtBtnImg}
        onClick={onGroupIconClick}
      />
    </div>
  );
};

export default NextBtn;
