import React from "react";
import backimg from "../../../assets/images/select-back.gif";
import "./SelectTxt.css";

const SelectTxt = ({ text, answer1, answer2, shift1, shift2, onNext }) => {

  const handleNextClick1 = () => {
    if (shift1 === 2) {
      onNext();
      onNext() 
    } else {
      onNext();
    }
  };

  const handleNextClick2 = () => {
    if (shift2 === 2) {
      onNext(); 
      onNext(); 
    } else {
      onNext();
    }
  };

  return (
    <div className="simul-412">
      <img src={backimg} alt="배경" className="background" />
      <div className="div003">{text}</div>
      <div className="divforselect">
        <div className="buttonslight001"onClick={handleNextClick1}>
          <div className="button-wrapper001" >
            <div className="div">{answer1}</div>
          </div>
        </div>
        {answer2 !== null && (
          <div className="buttonslight002"onClick={handleNextClick2}>
            <div className="button-wrapper001" >
              <div className="div">{answer2}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SelectTxt;
