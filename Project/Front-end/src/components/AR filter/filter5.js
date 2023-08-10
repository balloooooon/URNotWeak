import React from "react";
import "./filter5.css";
import NextBtn from "components/SimulationForm/NextBtn";

function ExternalPage({ onNext }) {
  return (
    <>
      <NextBtn />
      <div className="centered-and-scaled">
        <object data="/filter5.html" />
      </div>
    </>
  );
}

export default ExternalPage;
