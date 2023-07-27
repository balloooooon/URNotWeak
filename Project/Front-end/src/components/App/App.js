import React from "react";
import "./App.css";
import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";

import Home from "pages/Homepage/HomePage"
import Simulation from "pages/Simulation/Simulation"


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="simul" element={<Simulation />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
