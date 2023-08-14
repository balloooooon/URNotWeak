import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import ImageSlider from "components/common/BeforeAfterImg/ImageSlider";
import "./aiUpload.css";
import axios from "axios";
import Loading from "components/common/Loading/Loading";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
// import UserInteraction from "components/SimulationForm/BeforeAfterImg/UserInteraction";

export const AI = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState(null);
  const [loading, setLoading] = useState(false);
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
    setLoading(true);
    // AI Server 주소 가져오기
    let apiUrl;
    const urlResult = await axios({
      method: "GET",
      url: "https://urnotweak.site:8589/aiurl",
      responseType: "type",
    });
    apiUrl = urlResult.data;
    apiUrl = apiUrl + "/AI";
    console.log("AI Server 주소 : ", apiUrl);

    // forData 생성
    const formData = new FormData();
    formData.append("file", selectedFiles);

    // AI Server로 업로드한 이미지 전송
    const result = await axios({
      method: "POST",
      url: apiUrl,
      headers: {
        "Content-Type": "multipart/form-data",
      },
      data: formData,
      responseType: "blob",
    })
      .then((response) => {
        console.log(response);
        const result = URL.createObjectURL(response.data);
        navigate("/ai/result", {
          state: {
            aiImg: result,
          },
        });
      })
      .catch((Error) => {
        toast.error(
          <div>
            얼굴을 인식하지 못했습니다!
            <br />
            얼굴이 잘 보이는 사진으로 업로드 해주세요.
          </div>
        );
      });
    setLoading(false);
  };

  return (
    <div className="upload">
      <ToastContainer
        className="toast"
        autoClose={30000}
        position="top-right" // 알람 위치 지정
        hideProgressBar={true} // 진행시간바 숨김
        closeOnClick // 클릭으로 알람 닫기
        rtl={false} // 알림 좌우 반전
        pauseOnFocusLoss // 화면을 벗어나면 알람 정지
        draggable // 드래그 가능
        pauseOnHover // 마우스를 올리면 알람 정지
        theme="light"
        // limit={1} // 알람 개수 제한
      />

      {loading ? <Loading /> : null}
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
