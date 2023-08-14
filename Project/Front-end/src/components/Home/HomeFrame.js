import React from "react";
import { Link } from "react-router-dom";
import { Content } from './Content/Content'
import "./HomeFrame.css";
import NxtBtnImg from "assets/images/arrow right.svg";

export const Frame = ({ backimage, text1, text2, link }) => {

    return (

        <div className={`homeFrame`}>
            <div className={`overlap `}>
                <div className="rectangle big-tophalf" />
                <img className={`big-tophalf img ${backimage}`}></img>
                <div className={`group`}>
                    <div className="tx-b">{text2}</div>
                    <div className="title tx-rt">{text1}</div>
                </div>
                {text2==""?
                    <></>:
                    <Link to={`/${link}`} className="link-btn" style={{ textDecoration: "none" }}>
                        <div className="subtitle tx-rt">{text2} 확인하기</div> 
                        <img className="nxtbtn" alt={text2} src={NxtBtnImg}/>
                    </Link>
                }
            </div>
            
            {backimage==="home5"?<Content></Content>: <></>}
        </div>
    );
};
