import React from "react";
import { Nav } from "components/Nav/Nav";
import { StartForm } from "components/SimulationForm/StartForm/StartForm"

export const Simulation = () => {
    return (
        <div>
            <Nav className="nav-instance" />
            <StartForm />
        </div>
    );
};

export default Simulation;