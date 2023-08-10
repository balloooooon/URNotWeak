import './Footer.css'
import { Link } from "react-router-dom";
import NxtBtnImg from "assets/images/arrow right.svg";

export const Footer = ({goTop}) => {
    return (
        <div className={`homeFrame`}>
            <div className={`big-top black `}/>
            <div className="group p-ftop">
                <div className="tx-xb">당신은</div>
                <div className="tx-xb">약하지 않습니다.</div>
            </div>
            <div className='group p-middle'>
                <div className='item'>
                    <div className='tx-r'>Site Map</div>
                    <div></div><Link to={`simul`} className="tx-t" style={{ textDecoration: "none" }}>__ simulation</Link>
                    <div></div><Link to={`test`} className="tx-t" style={{ textDecoration: "none" }}>__ test</Link>
                    <div></div><Link to={`ai/upload`} className="tx-t" style={{ textDecoration: "none" }}>__ simulation</Link>
                </div>
                <div className='item'>
                    <div className='tx-r'>관련 사이트</div>
                    <div></div><a target='_blank' href='https://www.drugfree.or.kr/' className="tx-t" style={{ textDecoration: "none" }}>한국마약퇴치운동본부</a>
                    <div></div><a target='_blank' href='http://gdarc.org/' className="tx-t" style={{ textDecoration: "none" }}>다르크센터</a>
                </div>
            </div>
            <div className='up-btn'>
                <img className="nxtbtn" src={NxtBtnImg} onClick={goTop}/>
            </div>
            <div className="bottom-footer">
                <div className='left'>
                    <div className="tx-b">URNotWeek</div>
                    <div className="long-line"></div>
                    <div className="tx-t">대한민국 마약 근절 프로젝트</div>
                </div>
                <div className='right'>
                    <div className="tx-s">연락처 : 02-3429-5100</div>
                    <div className="tx-s">이메일 : ssafy@ssafy.com</div>
                    <div className="tx-s">CopyRight &copy; SAMSUMG SW ACADEMY</div>
                </div>
            </div>
        </div>
    );
};
