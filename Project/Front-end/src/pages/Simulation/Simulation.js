import React from "react";
import "./Simulation.css";
import { Nav } from "components/Nav/Nav";
import SelectForm from "components/SimulationForm/SelectForm/SelecctForm";
export const Simulation = () => {
  return (
    <div className="simul">
      <Nav />
      <SelectForm />
    </div>
  );
};

export default Simulation;
