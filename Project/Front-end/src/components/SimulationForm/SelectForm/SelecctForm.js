import React, { useState } from "react";
import SelectBox from "./SelectBox";
import StartAnd from "../StartForm/StartAnd";
import ResultForm from "../ResultForm/ResultForm";
import Nav from "components/Nav/Nav";
import "./SelectForm.css";

const SelectForm = () => {
  const [selectedIndex, setSelectedIndex] = useState(null);
  const handleThumbClick = (index) => {
    setSelectedIndex(index);
  };

  return (
    <div className="simul-">
      {selectedIndex === 1 ? (
        <StartAnd />
      ) : (
        <div>
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
