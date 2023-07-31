import React from "react";
import "./Simulation.css";
import { Nav } from "components/Nav/Nav";
import StartAnd from "components/SimulationForm/StartForm/StartAnd";
import NextBtn from "components/SimulationForm/NextBtn";
import StartTxt from "components/SimulationForm/StartForm/StartTxt";
import SelectForm from "components/SimulationForm/SelectForm/SelecctForm";
import VideoForm from "components/SimulationForm/StoryForm/VideoForm";
export const Simulation = () => {
  return (
    <div>
      {/* <Nav className="nav-instance" /> */}
      <SelectForm />
      {/* <StartAnd /> */}
      {/* <VideoForm /> */}
    </div>
  );
};

export default Simulation;
