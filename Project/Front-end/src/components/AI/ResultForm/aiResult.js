import React from "react";
import {useNavigate} from "react-router-dom";
import { BottomMenu } from "components/common/BottomMenu/BottomMenu";
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
      <BottomMenu></BottomMenu>
    </div>
  );
};
    
export default AI;
