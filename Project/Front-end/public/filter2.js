const camera = document.querySelector('#camera');
const playBtn = document.querySelector('.playBtn');
const canvas = document.querySelector('.main');
const text = document.querySelector('.stuff');
const ctx = canvas.getContext('2d');

const distortionCanvas = document.createElement('canvas');
const distortionCtx = distortionCanvas.getContext('2d');

const effectCanvas = document.createElement('canvas');
const effectCtx = effectCanvas.getContext('2d');

const MAX_DELAY = 100;

const OPTIONS = {
  delay: 70,
  delayRandom: 0.7,
  glitchFrequency: 74,
  glitchMultiply: true,
  specular: false,
  colorEffect: true };


// const gui = new dat.GUI();
// gui.add(OPTIONS, 'delay').min(0).max(MAX_DELAY).step(1);
// gui.add(OPTIONS, 'delayRandom').min(0).max(1).step(0.001);
// gui.add(OPTIONS, 'glitchFrequency').min(0).max(100).step(1);
// gui.add(OPTIONS, 'glitchMultiply');
// gui.add(OPTIONS, 'specular');
// gui.add(OPTIONS, 'colorEffect')

class Fifo {
  constructor(max) {
    this.max = max;
    this.els = [];
  }

  push(el) {
    this.els.unshift(el);
    if (this.els.length > this.max) {
      this.els.pop();
    }
  }

  at(i) {
    return this.els[i];
  }

  size() {
    return Math.min(this.els.length, this.max);
  }}


const pastFrames = new Fifo(MAX_DELAY);

const STATE = {
  FRAME_COUNT: 0,
  W: window.innerWidth,
  H: window.innerHeight,
  X: window.innerWidth / 2,
  Y: window.innerHeight / 2,
  MIN: Math.min(window.innerWidth, Math.innerHeight),
  P_DOWN: false,
  progress: 0,
  COLOR_BG: 'hsl(4, 90, 58)' };


function init() {
  const tl = new TimelineMax({ repeat: -1, yoyo: true });
  tl.fromTo(STATE, 1, { progress: 0 }, { progress: 1, ease: Power0.easeNone });
  tl.fromTo(STATE, 0.5, { COLOR_BG: 'hsl(4, 90, 58)' }, { COLOR_BG: 'hsl(208, 79, 51)', ease: Power0.easeNone }, 0);
  tl.fromTo(STATE, 0.5, { COLOR_BG: 'hsl(208, 79, 51)' }, { COLOR_BG: 'hsl(49, 98, 60)', ease: Power0.easeNone }, 0.5);
}

function update(t) {

}

function draw() {
  clear();

  drawCamera();
  grayscale();
  delay();

  distortionCtx.globalCompositeOperation = 'destination-out';
  distortionCtx.drawImage(effectCanvas, 0, 0);
  distortionCtx.globalCompositeOperation = 'source-over';

  ctx.fillRect(0, 0, STATE.W, STATE.H);
  ctx.globalCompositeOperation = 'difference';
  ctx.drawImage(effectCanvas, 0, 0);

  glitch();
  ctx.globalCompositeOperation = 'source-over';
}

function delay() {
  frameData = effectCtx.getImageData(0, 0, STATE.W, STATE.H);

  const delayRandom = OPTIONS.delayRandom * Math.random() || 1;
  const delay = OPTIONS.delay - 1;
  const frame = pastFrames.at(~~(delay * delayRandom)) || frameData;
  effectCtx.putImageData(frame, 0, 0);

  pastFrames.push(frameData);
}

function clear() {
  ctx.fillStyle = STATE.COLOR_BG;
  ctx.clearRect(0, 0, STATE.W, STATE.H);
  distortionCtx.clearRect(0, 0, STATE.W, STATE.H);
  effectCtx.clearRect(0, 0, STATE.W, STATE.H);
}

function grayscale() {
  const imageData = distortionCtx.getImageData(0, 0, STATE.W, STATE.H);
  const { data } = imageData;

  //https://www.html5canvastutorials.com/advanced/html5-canvas-grayscale-image-colors-tutorial/
  for (var i = 0; i < data.length; i += 4) {
    var brightness = 0.34 * data[i] + 0.5 * data[i + 1] + 0.16 * data[i + 2];
    data[i] = brightness;
    data[i + 1] = brightness;
    data[i + 2] = brightness;
    data[i + 3] = brightness;
  }

  effectCtx.putImageData(imageData, 0, 0);
}

function drawCamera() {
  const cH = STATE.H; // 이미지의 높이를 화면 높이와 동일하게 설정
  const cW = camera.videoWidth * (cH / camera.videoHeight); // 이미지의 너비를 가로 비율에 맞춰 계산
  const offset = (cW - STATE.W) / 2;

  if (OPTIONS.specular) {
    distortionCtx.drawImage(
      camera,
      camera.videoWidth / 2, 0,
      camera.videoWidth / 2, camera.videoHeight,
      STATE.W / 2, -offset,
      STATE.W / 2, cH);

    distortionCtx.save();
    flipX(distortionCtx);
    distortionCtx.drawImage(
      camera,
      camera.videoWidth / 2, 0,
      camera.videoWidth / 2, camera.videoHeight,
      0, -offset,
      STATE.W / 2, cH);

    distortionCtx.restore();
  } else {
    distortionCtx.drawImage(
      camera,
      0, 0,
      camera.videoWidth, camera.videoHeight,
      -offset, 0,
      cW, cH);

  }
}

function flipX(c) {
  c.setTransform(-1, 0, 0, 1, STATE.W / 2, 0);
}

function glitch() {
  if (OPTIONS.glitchFrequency === 0) return;

  if (OPTIONS.glitchMultiply) {
    ctx.globalCompositeOperation = 'multiply';
  }

  if (~~(STATE.FRAME_COUNT * (OPTIONS.glitchFrequency / 100) % 2)) {
    return;
  }

  ctx.drawImage(canvas,
  Math.random() * STATE.W, Math.random() * STATE.H,
  Math.random() * STATE.W, Math.random() * STATE.H,
  Math.random() * STATE.W, Math.random() * STATE.H,
  Math.random() * STATE.W, Math.random() * STATE.H);

}


window.addEventListener('resize', onResize);

window.addEventListener('mouseup', onPointerUp);
window.addEventListener('touchend', onPointerUp);
window.addEventListener('mousedown', onPointerDown);
window.addEventListener('touchstart', onPointerDown);
window.addEventListener('mousemove', ({ clientX, clientY }) => onPointerMove(clientX, clientY));
window.addEventListener('touchmove', ({ touches: [{ pageX, pageY }] }) => onPointerMove(pageX, pageY));


function start() {
  onResize();
  init();
  camera.onloadedmetadata = function() {
    camera.play();
    TweenMax.to(text, 0.3, { opacity: 0, ease: Power0.easeNone });
    requestAnimationFrame(loop);
  };
}

function loop(t) {
  draw();
  update(t);
  ++STATE.FRAME_COUNT;
  requestAnimationFrame(loop);
}

function onResize() {
  canvas.width = distortionCanvas.width = effectCanvas.width = STATE.W = window.innerWidth;
  canvas.height = distortionCanvas.height = effectCanvas.height = STATE.H = window.innerHeight;
  STATE.MAX = Math.max(STATE.W, STATE.H);
  STATE.MIN = Math.min(STATE.W, STATE.H);
}

function onPointerMove(x, y) {
  STATE.X = x;
  STATE.Y = y;
}

function onPointerDown() {
  STATE.P_DOWN = true;
}

function onPointerUp() {
  STATE.P_DOWN = false;
}

///
navigator ?
navigator.mediaDevices.getUserMedia({ video: true }).
then(withCamera).
catch(withVideo) :
withVideo();

function withCamera(stream) {
  camera.srcObject = stream;
  start();
}

function withVideo() {
  camera.src = 'https://res.cloudinary.com/dzckyfkls/video/upload/v1547465813/Smoke_Out_Of_Pipe.mp4';
  camera.crossOrigin = 'anonymous';
  start();
}

playBtn.onclick = function () {
  camera.play();
  TweenMax.to(text, 0.3, { opacity: 0, ease: Power0.easeNone });
};

// export default App;