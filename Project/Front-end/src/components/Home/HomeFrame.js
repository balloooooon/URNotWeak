import React from "react";
import { Link } from "react-router-dom";
import "./HomeFrame.css";
import NxtBtnImg from "assets/images/arrow right.svg";

export const Frame = ({ backimage, text1, text2, link, position }) => {

    return (
        <div className={`homeFrame`}>
            <div className={`overlap `}>
                <div className="rectangle big-top" />
                <img className={`big-top img ${backimage}`}></img>
                <div className={`group ${position}`}>
                    <div className="title tx-b">
                        {text1}
                    </div>
                    {text2==""?
                        <></>
                        :
                        <Link to={`/${link}`} className="overlap-group" style={{ textDecoration: "none" }}>
                            <div className="home-btn">
                                <div className="subtitle tx-t">{text2}</div> 
                                <img className="nxtbtn" alt={text2} src={NxtBtnImg}/>
                            </div>
                        </Link>
                    }
                </div>
            </div>
        </div>
    );
};
