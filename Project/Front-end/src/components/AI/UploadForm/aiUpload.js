import React, {useState, useEffect} from "react";
import {useNavigate} from "react-router-dom";
import ImageSlider from "components/common/BeforeAfterImg/ImageSlider";
import './aiUpload.css'

export const AI = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const navigate = useNavigate();

  // 사진 올렸을 때 실행
  const onUpload = (e) => {
    // 에러처리
    console.log(e.target.files);
    if(e.target.files.length == 0) return;

    // file -> url 변환
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.readAsDataURL(file);

    return new Promise((resolve) => { 
        reader.onload = () => {	
            setImageSrc(reader.result || null); // 파일의 컨텐츠
            resolve();
        };
    });
  }

  // 생성하기 눌렀을때
  const createResult = () => {
    // 사진 보내서 결과 받아오기 과정 필요


    navigate('/ai/result');
  }


  return (
    <div className="upload">
      <div>
        <p className="st">예시</p>
        <ImageSlider/>
      </div>
      <div className="split">
        <label htmlFor="file">
          <div className="btn">파일 업로드하기</div>
        </label>
        <input 
          name="file"
          id="file"
          accept="image/*" 
          multiple type="file"
          onChange={e => onUpload(e)}>
        </input>
        <div className="btn" onClick={createResult}>사진 생성하기</div>
      </div>
      <img 
        className="uploadimg"
        src={imageSrc} 
        />
    </div>
  );
};

export default AI;
