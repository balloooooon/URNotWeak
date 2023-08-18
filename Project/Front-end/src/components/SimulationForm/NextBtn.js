
import NxtBtnImg from "../../assets/group-45.svg";
import "./NextBtn.css";
const NextBtn = ({ onClick }) => {
  return <img className="next-btn" alt="" src={NxtBtnImg} onClick={onClick} />;

};

export default NextBtn;
