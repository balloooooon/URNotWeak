import React from "react";
import "./Simulation.css";
import { Nav } from "components/Nav/Nav";
import SelectForm from "components/SimulationForm/SelectForm/SelectForm";

import CardForm from "components/SimulationForm/StoryForm/CardForm";

export const Simulation = () => {
  return (
    <div className="simul">
      <Nav />
      {/* <SelectForm /> */}
      <CardForm initialFlag={1} />
    </div>
  );
};

export default Simulation;
