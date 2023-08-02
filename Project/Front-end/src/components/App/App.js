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
import AI from 'pages/AI/AI'

scrollbar.init(document.querySelector('#smooth-scroll'));

function App() {
  return (
    <div>
      <div className="display">
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="simul" element={<Simulation />} />
            <Route path="ai" element={<AI/> } />
          </Routes>
        </BrowserRouter>
      </div>
    </div>
  );
}

export default App;
