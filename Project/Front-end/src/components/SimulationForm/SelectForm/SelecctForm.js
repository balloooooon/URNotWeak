import React, { useState } from "react";
import SelectBox from "./SelectBox";
import StartAnd from "../StartForm/StartAnd";
import StartTxt from "../StartForm/StartTxt";
import VideoForm from "../StoryForm/VideoForm";
import "./SelectForm.css";

const SelectForm = () => {
  const [selectedIndex, setSelectedIndex] = useState(null);
  const handleThumbClick = (index) => {
    setSelectedIndex(index);
  };

  return (
    <div className="simul-">
      {selectedIndex === 2 ? (
        <StartAnd selectedIndex={selectedIndex} />
      ) : selectedIndex === 1 ? (
        <VideoForm selectedIndex={selectedIndex} />
      ) : selectedIndex === 3 ? (
        <StartTxt selectedIndex={selectedIndex} />
      ) : (
        <div className="simul-form">
          <div className="select-parent">
            <div className="simul-txt tx-b">시뮬레이션</div>
            <div className="simul-txt2 tx-t">시나리오 선택</div>
          </div>
          <SelectBox ThumbClick={handleThumbClick} />
        </div>
      )}
    </div>
  );
};

export default SelectForm;
