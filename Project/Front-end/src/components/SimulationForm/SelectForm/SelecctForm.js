import React from "react";
import SelectBox from "./SelectBox";
import "./SelectForm.css";

const SelectForm = () => {
  return (
    <div className="simul-">
      <div className="simul-child" />
      <div className="parent">
        <div className="div">시뮬레이션</div>
        <div className="div1">시나리오 선택</div>
        <SelectBox />
      </div>
    </div>
  );
};

export default SelectForm;
