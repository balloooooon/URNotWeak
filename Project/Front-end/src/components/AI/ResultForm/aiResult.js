import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
// import { BottomMenu } from "components/common/BottomMenu/BottomMenu";
import { Link } from "react-router-dom";
import { BottomMenu } from "components/common/BottomMenu/BottomMenu";

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
      <div className="btn tx-s" onClick={goUpload}>
        다시해보기
      </div>
      <BottomMenu></BottomMenu>
    </div>
  );
};

export default AI;
