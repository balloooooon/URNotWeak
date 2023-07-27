import React from "react";
import { Menu } from "./Menu";
// import "./style.css";

export const Nav = () => {
    return (
        <div className="nav">
            <div className="element">약해지지마</div>
            <Menu className="m-menu" color="black" />
        </div>
    );
};
