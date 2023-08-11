import React from 'react';
import NextBtn from 'components/SimulationForm/NextBtn';
import './filter3.css'; 

function ExternalPage({ onNext }) {
  return (
    <>
    <div className='filter-container'>
      <object data="/filter3.html" />
    </div>
    {onNext && <NextBtn onClick={onNext} />}
    </>
  );
}

export default ExternalPage;