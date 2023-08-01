import { useCallback } from "react";
import backimg from "../../../assets/images/select-back.gif";
import "./SelectTxt.css";
const SelectTxt = () => {
  return (
    <div className="simul-4">
      <div className="background-parent">
        <img src={backimg} alt="배경" className="background" />
        <div className="group-child" />
      </div>

      <div className="buttonslight-parent">
        <div className="buttonslight">
          <div className="button-wrapper">
            <div className="div">아기를 위해 참는다.</div>
          </div>
        </div>
        <div className="buttonslight1">
          <div className="button-wrapper">
            <div className="div">
              <p className="p">너무 힘들어서 견딜 수 없어 ..</p>
              <p className="p">아기에게는 미안하지만</p>
              <p className="p"> 마약을 한다.</p>
            </div>
          </div>
        </div>
        <div className="div3">
          <p className="p">마약에 중독되어 살던 중</p>
          <p className="p">임신 사실을 확인했다.</p>
          <p className="p">&nbsp;</p>
          <p className="p">약을 계속하면</p>
          <p className="p">아기가 잘못될 수도 있다는데..</p>
        </div>
      </div>
    </div>
  );
};

export default SelectTxt;
