import React, { useState } from "react";
import SelectBox from "./SelectBox";
import StartAnd from "../StartForm/StartAnd";
import StartTxt from "../StartForm/StartTxt";
import "./SelectForm.css";

const SelectForm = () => {
  const [selectedIndex, setSelectedIndex] = useState(null);
  const handleThumbClick = (index) => {
    setSelectedIndex(index);
  };

  return (
    <div className="simul-">
      {selectedIndex === 1 || selectedIndex === 2 ? (
        <StartAnd selectedIndex={selectedIndex} />
      ) : selectedIndex === 3 ? (
        <StartTxt selectedIndex={selectedIndex} />
      ) : (
        <div className="simul-form">
          <div className="select-parent">
            <div className="simul-txt">시뮬레이션</div>
            <div className="simul-txt2">시나리오 선택</div>
          </div>
          <SelectBox ThumbClick={handleThumbClick} />
        </div>
      )}
    </div>
  );
};

export default SelectForm;
