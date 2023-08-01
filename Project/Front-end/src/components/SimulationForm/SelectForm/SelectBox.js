import React from "react";
import Thumnail1 from "../../../assets/images/thumnail1.gif";
import Thumnail2 from "../../../assets/images/thumnail2.gif";
import Thumnail3 from "../../../assets/images/thumnail3.gif";
import "./SelectBox.css";

const SelectBox = ({ ThumbClick }) => {
  const ThumbnailImg = [Thumnail1, Thumnail2, Thumnail3];
  const handleThumbClick = (index) => {
    ThumbClick(index);
  };
  return (
    <div className="box">
      {ThumbnailImg.map((thumbnail, index) => (
        <div key={index} className={`ImgBox${index + 1}`}>
          <img
            className={`Thumnail${index + 1}`}
            alt=""
            src={thumbnail}
            onClick={() => handleThumbClick(index)}
          />
        </div>
      ))}
    </div>
  );
};

export default SelectBox;
