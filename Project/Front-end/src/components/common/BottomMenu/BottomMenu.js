import React from "react";
import { Link } from "react-router-dom";
import './BottomMenu.css';

export const BottomMenu = () => {
    return (
      <div className="bottom-menu">
      <div className="div2225">추천</div>
      <div
        className="buttonslight2222"
        // onClick={onButtonsLightContainer2Click}
      >
        <div className="button-wrapper222">
          <Link to="/" className="div222"  style={{ textDecoration: "none" }}>메인 페이지로</Link>
        </div>
      </div>
      <div className="group-container222">
        <div className="group-parent222">
          <div className="rectangle-parent222">
            <div className="group-item222" />
            <div className="group-inner222" />
            <div className="ai">시뮬레이션</div>
          </div>
          <div className="rectangle-group222">
            <div className="group-item222" />
            <div className="group-inner222" />
            <div className="ai222">AI</div>
          </div>
          <div className="rectangle-container222">
            <div className="group-item222" />
            <div className="group-inner222" />
            <div className="ai222">채팅</div>
          </div>
          <div className="rectangle-parent2221">
            <div className="group-item222" />
            <div className="group-inner222" />
            <div className="ai222">...</div>
          </div>
        </div>
      </div>
    </div>
    );
};