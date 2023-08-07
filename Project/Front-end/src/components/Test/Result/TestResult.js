import React from "react";
import {useLocation, useNavigate} from "react-router-dom";

export const AI = () => {
  const searchParams = new URLSearchParams(useLocation().search);
  const score = searchParams.get('score');
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
      <div className="btn" onClick={goUpload}>{score}테스트결과페이지</div>
    </div>
  );
};
    
export default AI;
