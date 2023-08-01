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

  if (selectedIndex === 1) {
    return <StartAnd />;
  } else if (selectedIndex === 2) {
    return <ResultForm />;
  }

  return (
    <div className="simul-">
      <Nav />
      <div className="parent">
        <div className="simul-txt">시뮬레이션</div>
        <div className="simul-txt2">시나리오 선택</div>
      </div>
      <SelectBox ThumbClick={handleThumbClick} />
    </div>
  );
};

export default SelectForm;
