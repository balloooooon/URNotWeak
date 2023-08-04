import React, { useState } from "react";
import backimg from "../../../assets/images/select-back.gif";
import "./SelectTxt.css";

const SelectTxt = ({ text, answer1, answer2, onNext }) => {
  return (
    <div className="simul-412">
      <img src={backimg} alt="배경" className="background" />
      <div className="div003">{text}</div>
      <div className="divforselect">
        <div className="buttonslight001">
          <div className="button-wrapper001" onClick={onNext}>
            <div className="div">{answer1}</div>
          </div>
        </div>
        {answer2 !== null && (
          <div className="buttonslight002">
            <div className="button-wrapper001" onClick={onNext}>
              <div className="div">{answer2}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SelectTxt;
