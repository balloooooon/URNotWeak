import React from "react";
import { Link } from "react-router-dom";
import './Menu.css';

export const MenuBar = ({closebtn}) => {
  return (
    <div className="bg">
        <div onClick={closebtn}>
            <svg
                className="close"
                fill="none"
                height="30"
                viewBox="0 0 24 24"
                width="30"
                xmlns="http://www.w3.org/2000/svg"
                >
                <path
                className="path"
                d="M17.5 8.055L16.44 7L12.25 11.195L8.055 7L7 8.055L11.195 12.25L7 16.44L8.055 17.5L12.25 13.31L16.44 17.5L17.5 16.44L13.31 12.25L17.5 8.055Z"
                fill="black"
                />
            </svg>
        </div>
        <div className="line-1"></div>
        <div className="menu-items">
            <div className="item"><Link to="/" className="text-wrapper" style={{ textDecoration: "none" }}>Home </Link></div>
            <div className="item"><Link to="/simul" className="text-wrapper" style={{ textDecoration: "none" }}>Simulation </Link></div>
            <div className="item"><Link to="/ai/upload" className="text-wrapper" style={{ textDecoration: "none" }}>AI </Link></div>
            <div className="item"><Link to="/test" className="text-wrapper" style={{ textDecoration: "none" }}>Test </Link></div>
            <div className="item"><Link to="/content/list" className="text-wrapper" style={{ textDecoration: "none" }}>Content </Link></div>
            <div className="item"><Link to="/ar" className="text-wrapper" style={{ textDecoration: "none" }}>AR Filter </Link></div>
            <div className="item"><Link to="/" className="text-wrapper" style={{ textDecoration: "none" }}>AboutUs </Link></div>
        </div>
    </div>
  );
};

export default MenuBar;
