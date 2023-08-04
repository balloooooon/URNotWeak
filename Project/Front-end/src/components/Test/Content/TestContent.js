import React from "react";
import {useNavigate} from "react-router-dom";

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
    </div>
  );
};
    
export default AI;
