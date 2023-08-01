import React from "react";
import "./Simulation.css";
import { Nav } from "components/Nav/Nav";
import StartAnd from "components/SimulationForm/StartForm/StartAnd";
import NextBtn from "components/SimulationForm/NextBtn";
import StartTxt from "components/SimulationForm/StartForm/StartTxt";
import SelectForm from "components/SimulationForm/SelectForm/SelecctForm";
import VideoForm from "components/SimulationForm/StoryForm/VideoForm";
import SelectTxt from "components/SimulationForm/StoryForm/SelectTxt";
export const Simulation = () => {
  return (
    <div>
      {/* <SelectTxt /> */}
      {/* <Nav className="nav-instance" /> */}
      <SelectForm />
      {/* <StartAnd /> */}
      {/* <VideoForm /> */}
    </div>
  );
};

export default Simulation;
