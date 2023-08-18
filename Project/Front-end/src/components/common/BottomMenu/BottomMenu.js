import React from "react";
import { Link } from "react-router-dom";
import './BottomMenu.css';

import ai_img from "../../../assets/images/rec_ai.png";
import ar_img from "../../../assets/images/rec_ar.png";
import simul_img from "../../../assets/images/rec_.png";

export const BottomMenu = () => {
    return (
      <div className="bottom-menu">
        
        <div className="line"></div>
        <p className="div2225 tx-s">추천</p>
        <div
          className="main-btn-wrapper"
          // onClick={onButtonsLightContainer2Click}
        >
          <div className="main-btn">
            <Link to="/" className="tx-s"  style={{ textDecoration: "none" }}>메인 페이지로</Link>
          </div>
        </div>
      <div className="menu-container">
        <div className="group-parent222">
          <div className="rectangle-parent222">
            <div className="group-item222" />
              <Link
                to="/simul"
                style={{ textDecoration: "none", color: "black" }}
              >
                <div
                  className="group-inner222"
                  style={{ backgroundImage: `url(${simul_img})` }}
                />
                <div className="ai222 tx-s-b">SIMULATION</div>
              </Link>
          </div>
          <div className="rectangle-group222">
            <div className="group-item222" />
              <Link
                to="/ai/upload"
                style={{ textDecoration: "none", color: "black" }}
              >
                <div
                  className="group-inner222"
                  style={{ backgroundImage: `url(${ai_img})` }}
                />
                <div className="ai222 tx-s-b">AI</div>
              </Link>
          </div>
          <div className="rectangle-container222">
            <div className="group-item222" />
              <Link
                to="/ar/filter2"
                style={{ textDecoration: "none", color: "black" }}
              >
                <div
                  className="group-inner222"
                  style={{ backgroundImage: `url(${ar_img})` }}
                />
                <div className="ai222 tx-s-b">AR</div>
              </Link>
          </div>
        </div>
      </div>
    </div>
    );
};