import React, { useState, useEffect } from "react";
import './ImageSlider.css'

const ImageSlider = ({ images }) => {
    const [currentIndex, setCurrentIndex] = useState(0);
  
    const nextImage = () => {
      setCurrentIndex((prevIndex) => (prevIndex === images.length - 1 ? 0 : prevIndex + 1));
    };
  
    // 타이머 기능 추가
    useEffect(() => {
      const slideShowInterval = setInterval(() => {
        nextImage();
      }, 4000); // 3초마다 이미지 변경
  
      // 컴포넌트가 언마운트될 때 타이머 정리
      return () => {
        clearInterval(slideShowInterval);
      };
    }, [currentIndex]); // currentIndex가 변경될 때마다 타이머 재설정
  
    return (
      <div className="exampleimg">
        <img className="imgitem" src={images[currentIndex]} alt={`Slide ${currentIndex}`} />
      </div>
    );
  };
  
  export default ImageSlider;