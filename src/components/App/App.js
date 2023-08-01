import React from "react";
import "./App.css";
import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";
import scrollbar from 'smooth-scrollbar'

import Home from "pages/Homepage/HomePage"
import Simulation from "pages/Simulation/Simulation"

scrollbar.init(document.querySelector('#smooth-scroll'));

function App() {
  return (
    <div className="display">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="simul" element={<Simulation />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
