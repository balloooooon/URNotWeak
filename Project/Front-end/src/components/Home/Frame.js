import React from "react";
import { Link } from "react-router-dom";
// import "./style.css";

export const Frame = ({ backimage, text1, text2, link }) => {
    return (
        <div className={`frame ${backimage}`}>
            <div className={`overlap `}>
                <div className="rectangle" />
                <div className="group">
                  <Link to={`/${link}`}>
                    <div className="overlap-group">
                        <p className="title btx">
                            {text1}
                        </p>
                        <div className="subtitle stx">{text2}</div>
                    </div>
                  </Link>
                </div>
            </div>
        </div>
    );
};
