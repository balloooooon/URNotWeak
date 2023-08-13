import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
// import { BottomMenu } from "components/common/BottomMenu/BottomMenu";
import { Link } from "react-router-dom";
import test_img from "../../../assets/images/rec_test.png";
import ar_img from "../../../assets/images/rec_ar.png";
import simul_img from "../../../assets/images/rec_.png";

import "./aiResult.css";
import { useLocation } from "react-router-dom";

export const AI = () => {
  const [imageSrc, setImageSrc] = useState(null);

  const navigate = useNavigate();

  const goUpload = () => {
    navigate("/ai/upload");
  };

  // navigator에서 보낸 마약전후 이미지 가져오기
  const location = useLocation();
  const aiFile = location.state.aiImg;

  useEffect(() => {
    setImageSrc(aiFile || null);
  }, []);

  return (
    <div className="upload">
      <div className="resultimg">
        <img className="uploadimg" src={imageSrc} />
      </div>
      <div className="btn" onClick={goUpload}>
        다시해보기
      </div>
      <div className="frame-div222">
        <div className="group-container222">
          <div className="group-parent222">
            <div
              className="rectangle-group222"
              // onClick={handleARClick}
            >
              <div className="group-item222" />
              <Link to="/test">
                <div
                  className="group-inner222"
                  style={{ backgroundImage: `url(${test_img})` }}
                />
                <div className="ai222">TEST</div>
              </Link>
            </div>
            <div className="rectangle-container222">
              <div className="group-item222" />
              <Link to="/ar/filter2">
                <div
                  className="group-inner222"
                  style={{ backgroundImage: `url(${ar_img})` }}
                />
              <div className="ai222">AR</div>
              </Link>
            </div>
            <div className="rectangle-parent222">
              <div className="group-item222" />
              <Link to="/simul">
                <div
                  className="group-inner222"
                  style={{ backgroundImage: `url(${simul_img})` }}
                />
                <div className="ai222">SIMULATION</div>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AI;
