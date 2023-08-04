import React from "react";
import { Link } from "react-router-dom";
import './BottomMenu.css';

export const BottomMenu = () => {
    return (
        <div className="dis">
            <div className="button-wrapper">
              <div className="div">추천</div>
            </div>
          <div className="buttonslight2">
            <div className="button-wrapper">
              <Link to="/" className="div" style={{ textDecoration: "none" }}>메인 페이지로</Link>
            </div>
          </div>
          <div className="group-container">
            <div className="group-parent">
                <Link to="/simul" className="overlap-group">
                    <div className="rectangle-parent">
                        <div className="group-item" />
                        <div className="group-inner" />
                        <div className="ai">시뮬레이션</div>
                    </div>
                </Link>
                <Link to="/ai/upload" className="overlap-group">
                    <div className="rectangle-group">
                        <div className="group-item" />
                        <div className="group-inner" />
                        <div className="ai">AI</div>
                    </div>
                </Link>
            </div>
          </div>
        </div>
    );
};