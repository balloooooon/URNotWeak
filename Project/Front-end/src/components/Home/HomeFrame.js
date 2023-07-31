import React from "react";
import { Link } from "react-router-dom";
import "./HomeFrame.css";

export const Frame = ({ backimage, text1, text2, link }) => {
    return (
        <div className={`frame`}>
            <div className={`overlap `}>
                <div className="rectangle big-top" />
                <img className={`big-top img ${backimage}`}></img>
                <div className="group">
                  <Link to={`/${link}`} className="overlap-group">
                    <p className="title btx">
                        {text1}
                    </p>
                    <div className="subtitle stx">{text2}</div>
                  </Link>
                </div>
            </div>
        </div>
    );
};
