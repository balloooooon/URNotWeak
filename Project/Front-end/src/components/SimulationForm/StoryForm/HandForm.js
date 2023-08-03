import React, { useRef } from "react";
import Draggable from "react-draggable";
import handback from "../../../assets/images/hand-back.png";
import handdirec from "../../../assets/images/handform-direc.svg";
import drug from "../../../assets/images/drug.png";

import "./HandForm.css";

const HandForm = ({ onNext }) => {
  const drugRef = useRef(null);

  const handleDragStart = () => {
    console.log("시작");
  };
const handleDrag=(ui)=>{
if(ui.changedTouches[0].clientY>=350){
{onNext()}}
}


  // const handleDrag = (e, ui) => {
  //   if (ui.deltaY >= 200) {
  //     console.log("성공");
  //   }
  // };

  return (
    <div className="hand-form">
      <div className="hand-parent">
        <img className="handbackimg" alt="" src={handback} />
        <div className="hand-group">
          <img className="hand-direc" alt="" src={handdirec} />
          {/* Draggable 컴포넌트로 감싸서 드래그 동작을 추가합니다. */}
          <Draggable
            axis="y"
            onStart={handleDragStart}
            onDrag={handleDrag}
            bounds={{ top: 0, bottom: 220 }}
            >
            {/* drug 이미지를 감싸는 div에 ref를 지정합니다. */}
            <div className="drug" ref={drugRef}>
              <img
                alt=""
                src={drug}
                onDrag={handleDrag} 
                onMouseDown={handleDragStart}
                onDragStart={(e) => e.preventDefault()} // 이미지 드래그의 기본 동작 방지
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
