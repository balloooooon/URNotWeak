import React, { useState, useEffect } from "react";
import './Loading.css'

import thumb1 from 'assets/images/contentThumnail01.png';
import thumb2 from 'assets/images/contentThumnail02.png';
import thumb3 from 'assets/images/contentThumnail03.png';

export const Loading = () => {
  const images = [thumb1, thumb2, thumb3]; // 이미지 배열
  const [currentImageIndex, setCurrentImageIndex] = useState(0); // 현재 이미지 인덱스

  useEffect(() => {
    const interval = setInterval(() => {
      // 다음 이미지 인덱스로 업데이트
      setCurrentImageIndex((prevIndex) => (prevIndex + 1) % images.length);
    }, 700); // 0.7초마다 이미지 변경

    return () => {
      clearInterval(interval); // 컴포넌트 언마운트 시 인터벌 해제
    };
  }, []);

  return (
    <div className="loading">
      <div className="wrapper">
        <img className="small-img" src={images[currentImageIndex]} alt={`Image ${currentImageIndex}`} />
        <p className="tx-s">로딩중입니다.</p>
      </div>
    </div>
  );
};

export default Loading;