import React, { useState } from 'react';
import './ImageSlider.css'; // 스타일 파일

export const ImageSlider = ({ images }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex === 0 ? 0 : prevIndex - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex === images.length - 1 ? images.length - 1 : prevIndex + 1));
  };

  return (
    <div className="image-slider">
      <div className="carousel">
        {images.map((image, index) => (
          <figure
            key={index}
            className={`image-wrapper slide ${index === currentIndex ? 'active' : ''}`}
            style={{
              transform: `translateX(-${currentIndex * 100}%)`, // 이미지 이동
            }}
          >
            <img src={image.contentImg}></img>
          </figure>
        ))}
      </div>

      <div className='left' onClick={handlePrev}><img></img></div>
      <div className='right' onClick={handleNext}><img></img></div>
      <div className="page-indicator">
        {currentIndex + 1} / {images.length}
      </div>
    </div>
  );
};

export default ImageSlider;





