import React from 'react';
import './filter4.css'; 
import NextBtn from 'components/SimulationForm/NextBtn';

function ExternalPage({ onNext }) {
  return (
    <>
    <div className="centered-and-scaled">
      <object data="/filter4.html" />
    </div>
    <NextBtn/>
    </>
  );
}

export default ExternalPage;