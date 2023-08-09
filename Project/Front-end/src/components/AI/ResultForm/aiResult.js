import React, {useState, useEffect} from "react";
import {useNavigate} from "react-router-dom";
import { BottomMenu } from "components/common/BottomMenu/BottomMenu";
import './aiResult.css'
import { useLocation } from 'react-router-dom';

export const AI = () => {
  const [imageSrc, setImageSrc] = useState(null);

  const navigate = useNavigate();

  const goUpload = () => {
    navigate('/ai/upload');
  }

  // const {state} = useLocation();	// 2번 라인
  
  //   console.log('user 값이 설정됨');
  // const { aiFile } = state;	// 3번 라인

   // 1. useLocation 훅 취득
  const location = useLocation();

  console.log(location.state);

  // 2. location.state 에서 파라미터 취득
  const aiFile = location.state.aiImg;
  console.log(typeof aiFile)
  console.log(aiFile);

    // const reader = new FileReader();
    // reader.readAsDataURL(aiFile);
  
    // reader.onload = () => {	
    //   setImageSrc(reader.result || null); // 파일의 컨텐츠
    // };
  useEffect(() => {
    setImageSrc(aiFile || null);
      
    },[])
    



  // 1. useLocation 훅 취득
  // const location = useLocation();

  // 2. location.state 에서 파라미터 취득
  // const aiFile = location.state.aiImg;


  return (
    <div className="upload">
      <img 
        className="uploadimg"
        src={imageSrc} 
        />
      <div className="btn" onClick={goUpload}>다시해보기</div>
      <BottomMenu></BottomMenu>
    </div>
  );
};
    
export default AI;
