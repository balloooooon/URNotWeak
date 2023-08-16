import React from "react";
import { Routes, Route} from "react-router-dom";
import { Nav } from "components/Nav/Nav";
import Filter1 from 'components/AR filter/filter1'
import Filter2 from 'components/AR filter/filter2';
import Filter3 from 'components/AR filter/filter3';
import Filter4 from 'components/AR filter/filter4';
import Filter5 from 'components/AR filter/filter5';
import Select from 'components/AR filter/Select';
// import AiUpload from 'components/AI/UploadForm/aiUpload';
// import AiResult from 'components/AI/ResultForm/aiResult';
import './AR.css'

export const AR = () => {

  return (
    <div className="display">
        <Nav className="nav-instance" />
        <div className="rectangle big-top" />
        <img className='big-top back'></img>

        <div className="content">
            <p className="main">AR 체험하기</p>
            <Routes>
                <Route path="/filter1" element={<Filter1 />}></Route>
                <Route path="/filter2" element={<Filter2 />}></Route>
                <Route path="/filter3" element={<Filter3 />}></Route>
                <Route path="/filter4" element={<Filter4 />}></Route>
                <Route path="/filter5" element={<Filter5 />}></Route>
                <Route path="/" element={<Select />}></Route>
                {/* <Route path="/result" element={<AiResult />}></Route> */}
            </Routes>
        </div>
    </div>
  );
};

export default AR;
