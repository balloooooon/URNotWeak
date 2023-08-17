import React from "react";
import {useNavigate} from "react-router-dom";
import NextBtn from "components/SimulationForm/NextBtn";
import loadimg from 'assets/images/testloading.gif'

export const AI = () => {
  const navigate = useNavigate();

  const goContent = () => {
    navigate('/test/content');
  }

  return (
    <div className="test-loading">
      <p className="test-main tx-b">취약성 테스트</p>s
      <img 
        className="uploadimg"
        src={loadimg} 
        />
      <NextBtn onClick={goContent}></NextBtn>
    </div>
  );
};
    
export default AI;
