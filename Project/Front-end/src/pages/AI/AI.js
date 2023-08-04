import React from "react";
import { Routes, Route} from "react-router-dom";
import { Nav } from "components/Nav/Nav";
import AiUpload from 'components/AI/UploadForm/aiUpload';
import AiResult from 'components/AI/ResultForm/aiResult';
import './AI.css'

import backimg from 'assets/images/ai-background.png'

export const AI = () => {

  return (
    <div className="display">
        <Nav className="nav-instance" />
        <div className="rectangle big-top" />
        <img className='big-top back' src={backimg}></img>

        <div className="content">
            <p className="main">전,후 사진 AI 체험하기</p>
            <Routes>
                <Route path="/upload" element={<AiUpload />}></Route>
                <Route path="/result" element={<AiResult />}></Route>
            </Routes>
        </div>
    </div>
  );
};

export default AI;
