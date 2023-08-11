import React from 'react';
import './filter4.css'; 
import NextBtn from 'components/SimulationForm/NextBtn';

function ExternalPage({ onNext }) {
  return (
    <>
    <div className='filter-container'>
      <object data="/filter4.html" />
    </div>
    {onNext && <NextBtn onClick={onNext} />}
    </>
  );
}

export default ExternalPage;