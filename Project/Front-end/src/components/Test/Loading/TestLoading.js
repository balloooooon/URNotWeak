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
    <div className="upload">
      <img 
        className="uploadimg"
        src={loadimg} 
        />
      <NextBtn onClick={goContent}></NextBtn>
    </div>
  );
};
    
export default AI;
