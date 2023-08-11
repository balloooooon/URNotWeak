import React from "react";
import { Routes, Route } from "react-router-dom";
import { Nav } from "components/Nav/Nav";
import AiUpload from 'components/AI/UploadForm/aiUpload';
import AiResult from 'components/AI/ResultForm/aiResult';
import ContentList from 'components/Content/List'


export const Content = () => {

  return (
    <div className="display">
        <Nav className="nav-instance" />
        <Routes>
            <Route path="/list" element={<ContentList />}></Route>
            <Route path="/detail" element={<AiResult />}></Route>
        </Routes>
    </div>
  );
};

export default Content;
