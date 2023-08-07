import React, { useState, useEffect } from "react";
import axios from 'axios';
import './ImageSlider.css'

const ImageSlider = () => {
  const [images, setImages] = useState([{drugBeforeImg: null, drugAfterImg: null}]);
  const [currentIndex, setCurrentIndex] = useState(0);
  
  // 사진 axios에서 들고오도록
  useEffect(() => {
    axios.get(process.env.REACT_APP_SERVER_API_URL+'/drugimg')
    .then((response)=>{saveImg(response.data)})
    .catch((Error)=>{console.log(Error)});

    // console.log(images);
  },[])

  const saveImg = (data) => {
    console.log('data!');
    console.log(data);

    setImages([...data]);
    console.log(images);
  }


  
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

  const nextImage = () => {
    // console.log("next"+currentIndex);
    setCurrentIndex((prevIndex) => (prevIndex === images.length - 1 ? 0 : prevIndex + 1));
  };


  return (
    <div className="exampleimg">
      <img className="imgitem" src={images[currentIndex].drugBeforeImg} alt={`Slide ${currentIndex}`} />
      <img className="imgitem" src={images[currentIndex].drugAfterImg} alt={`Slide ${currentIndex}`} />
    </div>
  );
};

export default ImageSlider;