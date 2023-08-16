import React, { useRef } from "react";
import Draggable from "react-draggable";
import handback from "../../../assets/images/hand-back.jpg";
import handdirec from "../../../assets/images/handform-direc.svg";
import drug from "../../../assets/images/drug.png";

import "./HandForm.css";

const HandForm = ({ onNext }) => {
  const drugRef = useRef(null);

  const handleDragStart = () => {
    console.log("시작");
  };

  const handleDrag = (ui) => {
    if (
      (ui.changedTouches &&
        ui.changedTouches[0] &&
        ui.changedTouches[0].clientY >= 390) ||
      (!ui.changedTouches && ui.clientY >= 390)
    ) {
      onNext();
    }
  };

  return (
    <div className="hand-form">
      <div className="hand-parent">
        <img className="handbackimg" alt="" src={handback} />
        <div className="hand-group">
          <img className="hand-direc" alt="" src={handdirec} />
          <Draggable
            axis="y"
            onStart={handleDragStart}
            onDrag={handleDrag}
            bounds={{ top: 0, bottom: 220 }}
          >
            <div className="drug" ref={drugRef}>
              <img
                alt=""
                src={drug}
                onDrag={handleDrag}
                onMouseDown={handleDragStart}
                onDragStart={(e) => e.preventDefault()}
              />
            </div>
          </Draggable>
          <div className="drag">드래그하여 내려주세요 !</div>
        </div>
      </div>
    </div>
  );
};

export default HandForm;
