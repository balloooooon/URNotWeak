import React from "react";
import {useNavigate} from "react-router-dom";
import { Link } from "react-router-dom";
import test_img from "../../../assets/images/rec_test.png";
import ar_img from "../../../assets/images/rec_ar.png";
import simul_img from "../../../assets/images/rec_.png";

// import { BottomMenu } from "components/common/BottomMenu/BottomMenu";
import './aiResult.css'

export const AI = () => {
  const navigate = useNavigate();

  const goUpload = () => {
    navigate('/ai/upload');
  }

  return (
    <div className="upload">
      <img 
        className="uploadimg"
        src={null} 
        />
      <div className="btn" onClick={goUpload}>다시해보기</div>
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
