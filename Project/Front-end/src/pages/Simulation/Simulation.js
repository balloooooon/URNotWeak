import React from "react";
import "./Simulation.css";
import { Nav } from "components/Nav/Nav";
import SelectForm from "components/SimulationForm/SelectForm/SelectForm";

import CardForm from "components/SimulationForm/StoryForm/CardForm";

const cards = [
  "https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card1.png",
  "https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card2.png",
  "https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card3.png",
  "https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card4.png",
  "https://ssafy-e204-bucket.s3.ap-northeast-2.amazonaws.com/E204/card/card5.png",
];

export const Simulation = () => {
  return (
    <div className="simul">
      <Nav />

      <SelectForm />
      {/* <CardForm cards={cards}/> */}
    </div>
  );
};

export default Simulation;
