import React from "react";
import { Routes, Route} from "react-router-dom";
import { Nav } from "components/Nav/Nav";
import TestContent from 'components/Test/Content/TestContent'
import TestLoading from 'components/Test/Loading/TestLoading'
import TestResult from 'components/Test/Result/TestResult'

import backimg from 'assets/images/test-background.jpg'
import './Test.css'

export const Test = () => {

  return (
    <div className="display">
        <Nav className="nav-instance" />
        <div className="rectangle7 big-top" />
        <img className='big-top back' src={backimg}></img>

        <div className="content">
            <p className="main">테스트</p>
            <Routes>
                <Route path="/" element={<TestLoading />}></Route>
                <Route path="/content" element={<TestContent />}></Route>
                <Route path="/result" element={<TestResult />} ></Route>
            </Routes>
        </div>
    </div>
  );
};

export default Test;
