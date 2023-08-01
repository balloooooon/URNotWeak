import React, { useState } from "react";
import SelectBox from "./SelectBox";
import StartAnd from "../StartForm/StartAnd";
import "./SelectForm.css";

const SelectForm = () => {
  const [selectedIndex, setSelectedIndex] = useState(null);
  const handleThumbClick = (index) => {
    setSelectedIndex(index);
  };

  if (selectedIndex == 1) {
    return <StartAnd />;
  }

  return (
    <div className="simul-">
      <div className="parent">
        <div className="simul-txt">시뮬레이션</div>
        <div className="simul-txt2">시나리오 선택</div>
      </div>
      <SelectBox ThumbClick={handleThumbClick} />
    </div>
  );
};

export default SelectForm;
