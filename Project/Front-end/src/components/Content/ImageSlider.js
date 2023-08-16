import React, { useState } from 'react';
import './ImageSlider.css'; // 스타일 파일

import asign from 'assets/images/arrow right.svg'

export const ImageSlider = ({ images }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [touchStartX, setTouchStartX] = useState(null);

  const handleTouchStart = (event) => {
    setTouchStartX(event.touches[0].clientX);
  };

  const handleTouchMove = (event) => {
    if (touchStartX === null) return;
    console.log("!!!!");
    const touchCurrentX = event.touches[0].clientX;
    const deltaX = touchStartX - touchCurrentX;

    console.log(deltaX);
    if (deltaX > 7) {
      handleNext();
      console.log(1)
    } else if (deltaX < -7) {
      handlePrev();
      console.log(2)
    }

    setTouchStartX(null);
  };

  const handlePrev = () => {
    setCurrentIndex((prevIndex) => (prevIndex === 0 ? 0 : prevIndex - 1));
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex === images.length - 1 ? images.length - 1 : prevIndex + 1));
  };


  return (
    <div
      className="image-slider"
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={() => setTouchStartX(null)}
    >
      <div className="carousel">
        {images.map((image, index) => (
          <figure
            key={index}
            className={`image-wrapper slide ${index === currentIndex ? 'active' : ''}`}
            style={{
              transform: `translateX(-${currentIndex * 100}%)`, // 이미지 이동
            }}
          >
            <img src={image.contentImg} alt={`Slide ${index}`} />
          </figure>
        ))}
      </div>

      <div className='left' onClick={handlePrev}><img src={asign}></img></div>
      <div className='right' onClick={handleNext}><img src={asign}></img></div>
      <div className="page-indicator">
        {currentIndex + 1} / {images.length}
      </div>
    </div>
  );
};

export default ImageSlider;





