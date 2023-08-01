import React from "react";
import { Menu } from "./Menu";
import { Link } from "react-router-dom";
import "./Nav.css";

export const Nav = () => {
  return (
      <div className="navbar">
        <Link to="/" className="element" style={{ textDecoration: "none" }}>약해지지마</Link>
        <Menu className="m-menu" color="white" />
      </div>
  );
};

export default Nav;
