import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import ImageSlider from "components/common/BeforeAfterImg/ImageSlider";
import "./aiUpload.css";
import axios from "axios";

export const AI = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState(null);
  const navigate = useNavigate();

  // 사진 올렸을 때 실행
  const onUpload = (e) => {
    // 에러처리
    console.log(e.target.files);
    if (e.target.files.length == 0) return;

    // file -> url 변환
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.readAsDataURL(file);

    return new Promise((resolve) => {
      reader.onload = () => {
        setImageSrc(reader.result || null); // 파일의 컨텐츠
        setSelectedFiles(file);
        resolve();
      };
    });
  };

  // 생성하기 눌렀을때
  const createResult = async () => {
    // AI Server 주소 가져오기
    let apiUrl;
    const result = await axios({
      method: "GET",
      url: "https://urnotweak.site:8589/aiurl",
      responseType: "type",
    });
    apiUrl = result.data;
    apiUrl = apiUrl + "/AI";
    console.log("AI Server 주소 : ", apiUrl);

    // forData 생성
    const formData = new FormData();
    formData.append("file", selectedFiles);

    // AI Server로 업로드한 이미지 전송
    axios({
      method: "POST",
      url: apiUrl,
      headers: {
        "Content-Type": "multipart/form-data",
      },
      data: formData,
      responseType: "blob",
    })
      .then((response) => {
        const result = URL.createObjectURL(response.data);
        navigate("/ai/result", {
          state: {
            aiImg: result,
          },
        });
      })
      .catch((Error) => {
        console.log(Error);
      });
  };

  return (
    <div className="upload">
      <div>
        <p className="st">예시</p>
        <ImageSlider />
      </div>
      <div className="split">
        <label htmlFor="file">
          <div className="btn">파일 업로드하기</div>
        </label>
        <input
          name="file"
          id="file"
          accept="image/*"
          multiple
          type="file"
          onChange={(e) => onUpload(e)}
        ></input>
        <div className="btn" onClick={createResult}>
          사진 생성하기
        </div>
      </div>
      <img className="uploadimg" src={imageSrc} />
    </div>
  );
};

export default AI;
