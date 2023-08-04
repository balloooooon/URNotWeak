import { FaceMesh } from "@mediapipe/face_mesh";
import React, { useRef, useEffect } from "react";
import * as cam from "@mediapipe/camera_utils";
import Webcam from "react-webcam";
import './style.css'

// import './script'
// import 'https://cdnjs.cloudflare.com/ajax/libs/gsap/2.0.2/TweenMax.min.js';
// import 'https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.5/dat.gui.min.js';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);


  return (
    <center>
      <canvas className="main"></canvas>
<video id="camera"></video>

<div className="stuff"></div>
<script src='https://cdnjs.cloudflare.com/ajax/libs/gsap/2.0.2/TweenMax.min.js'></script>
<script src= 'https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.5/dat.gui.min.js'></script>
<script src="./script.js"></script>

    </center>
  );
}

export default App;
