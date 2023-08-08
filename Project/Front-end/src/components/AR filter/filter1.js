import { FaceMesh } from "@mediapipe/face_mesh";
import React, { useRef, useEffect } from "react";
import * as cam from "@mediapipe/camera_utils";
import Webcam from "react-webcam";


function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  function drawImageOnLandmark(canvasCtx, image, landmark) {
    const imageWidth = image.width/9; // 이미지의 너비
    const imageHeight = image.height/9; // 이미지의 높이
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
    canvasCtx.globalAlpha = 0.4;

      
    }
  }

  useEffect(() => {
    const faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      },
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
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
        width: 1080,
        height: 720,
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
            width: 1080,
            height: 600,
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
            width: 1080,
            height: 600,
          }}
        />
      </div>
    </center>
  );
}

export default App;
