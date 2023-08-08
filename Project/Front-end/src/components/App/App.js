import React, {useEffect} from "react";
import "./App.css";
import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";
import scrollbar from 'smooth-scrollbar'

import Home from "pages/Homepage/HomePage"
import Simulation from "pages/Simulation/Simulation"
import AI from 'pages/AI/AI'
import AR from 'pages/AR/AR'
import Test from 'pages/Test/Test'

scrollbar.init(document.querySelector('#smooth-scroll'));

function App() {

  useEffect(() => {
    
    function setScreenSize() {
      const vh = window.innerHeight * 0.01;
      document.documentElement.style.setProperty('--vh', `${vh}px`);
    }

    setScreenSize();

    // 높이가 변경되었을때 다시 정의
    window.addEventListener('resize', () => setScreenSize());
  }, []);

  return (
    <div>
      <div className="display">
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="simul" element={<Simulation />} />
            <Route path="ai/*" element={<AI/> } />
            <Route path="ar/*" element={<AR />} />
            <Route path="test/*" element={<Test />} />
          </Routes>
        </BrowserRouter>
      </div>
    </div>
  );
}

export default App;
