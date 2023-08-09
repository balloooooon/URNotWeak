import { FaceMesh } from "@mediapipe/face_mesh";
import React, { useRef, useEffect } from "react";
import * as cam from "@mediapipe/camera_utils";
import Webcam from "react-webcam";

import ant2 from "../../assets/images/ant2.png"
import ant7 from "../../assets/images//ant7.png"; // 첫 번째 이미지의 경로를 입력해주세요.
import ant9 from "../../assets/images//ant9.png"; // 두 번째 이미지의 경로를 입력해주세요.
import ant10 from "../../assets/images//ant10.png"; // 세 번째 이미지의 경로를 입력해주세요.
import ant12 from "../../assets/images//ant12.png"; // 세 번째 이미지의 경로를 입력해주세요.

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  function drawImageOnLandmark(canvasCtx, image, landmark) {
    const imageWidth = image.width/2.2; // 이미지의 너비
    const imageHeight = image.height/2.2; // 이미지의 높이
    const imageX = landmark.x * canvasRef.current.width - imageWidth / 2; // 이미지의 x 좌표
    const imageY = landmark.y * canvasRef.current.height - imageHeight / 2; // 이미지의 y 좌표
    canvasCtx.drawImage(image, imageX, imageY, imageWidth, imageHeight);
    
  }

  function onResults(results) {
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      // 22번 랜드마크에 첫 번째 이미지를 넣습니다.
    const canvasElement = canvasRef.current;
    const canvasCtx = canvasElement.getContext("2d");
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.globalAlpha = 1;

      if (results.multiFaceLandmarks[0][346]) { //왼쪽 뺨
        const image1Element = new Image();
        image1Element.src = ant2;
        image1Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image1Element,
            results.multiFaceLandmarks[0][346]
            
          );
        };
      }
      if (results.multiFaceLandmarks[0][427]) { //왼쪽 턱쪽 뺨
        const image1Element = new Image();
        image1Element.src = ant10;
        image1Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image1Element,
            results.multiFaceLandmarks[0][427]
            
          );
        };
      }
      if (results.multiFaceLandmarks[0][299]) {
        const image1Element = new Image();
        image1Element.src = ant9;
        image1Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image1Element,
            results.multiFaceLandmarks[0][299]
            
          );
        };
      }
      if (results.multiFaceLandmarks[0][66]) { //오른쪽 이마
        const image1Element = new Image();
        image1Element.src = ant2;
        image1Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image1Element,
            results.multiFaceLandmarks[0][66]
            
          );
        };
      }
      if (results.multiFaceLandmarks[0][297]) { //왼쪽 이마
        const image1Element = new Image();
        image1Element.src = ant10;
        image1Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image1Element,
            results.multiFaceLandmarks[0][297]
            
          );
        };
      }
      if (results.multiFaceLandmarks[0][9]) { //가운데 이마
        const image1Element = new Image();
        image1Element.src = ant7;
        image1Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image1Element,
            results.multiFaceLandmarks[0][9]
            
          );
        };
      }
      if (results.multiFaceLandmarks[0][337]) { //가운데 이마2
        const image1Element = new Image();
        image1Element.src = ant10;
        image1Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image1Element,
            results.multiFaceLandmarks[0][337]
            
          );
        };
      }
      if (results.multiFaceLandmarks[0][200]) { // 입술아래
        const image1Element = new Image();
        image1Element.src = ant9
        image1Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image1Element,
            results.multiFaceLandmarks[0][200]
            
          );
        };
      }

      // 23번 랜드마크에 두 번째 이미지를 넣습니다.
      if (results.multiFaceLandmarks[0][187]) { //오른쪽 볼
        
        const image2Element = new Image();
        image2Element.src = ant7;
        image2Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image2Element,
            results.multiFaceLandmarks[0][187]
          );
        };
      }
      // if (results.multiFaceLandmarks[0][24]) { //왼쪽 눈아래
      //   const image2Element = new Image();
      //   image2Element.src = left_circle;
      //   image2Element.onload = () => {
      //     drawImageOnLandmark(
      //       canvasRef.current.getContext("2d"),
      //       image2Element,
      //       results.multiFaceLandmarks[0][24]
      //     );
      //   };
      // }
      // //24번 랜드마크에 세 번째 이미지를 넣습니다.
      // if (results.multiFaceLandmarks[0][253]) { //오른쪽 눈아래
      //   const image3Element = new Image();
      //   image3Element.src = right_circle;
      //   image3Element.onload = () => {
      //     drawImageOnLandmark(
      //       canvasRef.current.getContext("2d"),
      //       image3Element,
      //       results.multiFaceLandmarks[0][253]
      //     );
      //   };
      // }
      //24번 랜드마크에 세 번째 이미지를 넣습니다.
      if (results.multiFaceLandmarks[0][202]) { // 오른쪽 이마끝
        const image3Element = new Image();
        image3Element.src = ant9;
        image3Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image3Element,
            results.multiFaceLandmarks[0][202]
          );
        };
      }
      //24번 랜드마크에 세 번째 이미지를 넣습니다.
      if (results.multiFaceLandmarks[0][67]) { //오른쪽 턱끝
        const image3Element = new Image();
        image3Element.src = ant12;
        image3Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image3Element,
            results.multiFaceLandmarks[0][67]
          );
        };
      }
      if (results.multiFaceLandmarks[0][4]) { //코 조금아래
        const image3Element = new Image();
        image3Element.src = ant12;
        image3Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image3Element,
            results.multiFaceLandmarks[0][4]
          );
        };
      }
      if (results.multiFaceLandmarks[0][248]) { //코 조금위
        const image3Element = new Image();
        image3Element.src = ant10;
        image3Element.onload = () => {
          drawImageOnLandmark(
            canvasRef.current.getContext("2d"),
            image3Element,
            results.multiFaceLandmarks[0][248]
          );
        };
      }
    }
  }

  useEffect(() => {
    const faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      },
    });

    faceMesh.setOptions({
      maxNumFaces: 3,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults(onResults);

    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null
    ) {
      const camera = new cam.Camera(webcamRef.current.video, {
        onFrame: async () => {
          await faceMesh.send({ image: webcamRef.current.video });
        },
        width: 640,
        height: 480,
      });
      camera.start();
    }
  }, []);

  return (
    <center>
      <div className="App">
        {/* 웹캠 비디오를 표시합니다. */}
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
            filter: 'grayscale(30%) contrast(12) brightness(0.8)'
          }}
        />
        {/* 이미지를 그릴 캔버스입니다. */}
        <canvas
          ref={canvasRef}
          className="output_canvas"
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />
      </div>
    </center>
  );
}

export default App;
