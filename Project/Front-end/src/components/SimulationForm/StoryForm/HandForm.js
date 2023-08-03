import React, { useState } from "react";
import handback2 from "../../../assets/images/handback2.png";
import handback from "../../../assets/images/hand-back.png";
import handdirec from "../../../assets/images/handform-direc.svg";
import drug from "../../../assets/images/drug.png";

import "./HandForm.css";

const HandForm = ({ onNext }) => {
  const [handbackImg, setHandbackImg] = useState(handback);

  const handleDrag = (event) => {
    console.log("시작")
    const offsetY = event.clientY;
    if (offsetY >= 50) {
      setHandbackImg(handback2);
    } else {
      setHandbackImg(handback);
    }
  };

  return (
    <div className="hand-form">
      <div className="hand-parent">
        <img className="handbackimg" alt="" src={handbackImg} />
        <div className="hand-group">
          <img className="hand-direc" alt="" src={handdirec} />
          <img
            className="drug"
            alt=""
            src={drug}
            draggable="true"
            onDrag={handleDrag}
          />
          <div className="drag">드래그하여 내려주세요 !</div>
        </div>
      </div>
    </div>
  );
};

export default HandForm;
