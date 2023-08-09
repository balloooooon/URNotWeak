import React from "react";
import { Nav } from "components/Nav/Nav";
import { Link } from "react-router-dom";
import './NotFound.css'

import notfoundimg from 'assets/images/not-found.gif'

export const NotFound = () => {


  return (
    <div className="notfound">
      <Nav className="nav-instance" />
      <div className="wrapper">
        <div className="blank"></div>
        <img className="notfound-img" src={notfoundimg}/>
        <div className="txt">페이지를 찾을 수 없습니다.</div>
        <div className="btn">
          <Link to="/" className="div222"  style={{ textDecoration: "none" }}>메인 페이지로 돌아가기</Link>
        </div>
      </div>
      
    </div>
  );
};

export default NotFound;
