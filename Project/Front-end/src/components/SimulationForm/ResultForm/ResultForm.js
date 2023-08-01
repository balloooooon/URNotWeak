import instagram from "../../../assets/images/instagram.png"
import kakao from "../../../assets/images/kakao.png";
import link from "../../../assets/images/share.png";
import "./ResultForm.css";
const ResultForm = () => {
  return (
    <div className="result">
      <div className="frame-parent">
        <div className="api-wrapper">
          <div className="api-container">
            <p className="api">
              매년 (api)명의 중독자와 사망자가 발생하고 있습니다.
            </p>
            <p className="api">&nbsp;</p>
            <p className="api">당신은 안전할까요?</p>
          </div>
        </div>
        <div className="group">
          <div className="div2">
            <p className="api">{`이 시뮬레이션은 허구가 아닌 `}</p>
            <p className="api">사실기반의 체험입니다.</p>
          </div>
          <div className="frame-child" />
          <img className="frame-item" alt="" src="/vector-14.svg" />
        </div>
        <div className="container">
          <div className="div3">마약을하면..</div>
          <img className="frame-inner" alt="" src="/vector-141.svg" />
          <img className="icon" alt="" src="/-2@2x.png" />
          <div className="buttonslight-parent">
            <div className="buttonslight">
              <div className="button-wrapper">
                <div className="div">전, 후 사진 AI 체험하기</div>
              </div>
            </div>
            <div className="buttonslight1">
              <div className="button-wrapper">
                <div className="div">다른 스토리 확인하기</div>
              </div>
            </div>
          </div>
          <img className="vector-icon" alt="" src="/vector-142.svg" />
        </div>
        <div className="group-wrapper">
          <div className="group-div">
            <div className="div4">친구에게 공유하기</div>
            <img className="kakaotalk" alt="카톡공유이미지" src={kakao} />
            <img className="instagram" alt="인스타공유" src={instagram} />
            <div className="group-child" />
            <img className="share-url" alt="링크공유" src={link} />
          </div>
        </div>
        <div className="frame-div">
          <div className="div5">추천</div>
          <div
            className="buttonslight2"
            // onClick={onButtonsLightContainer2Click}
          >
            <div className="button-wrapper">
              <div className="div">메인 페이지로</div>
            </div>
          </div>
          <div className="group-container">
            <div className="group-parent">
              <div className="rectangle-parent">
                <div className="group-item" />
                <div className="group-inner" />
                <div className="ai">시뮬레이션</div>
              </div>
              <div className="rectangle-group">
                <div className="group-item" />
                <div className="group-inner" />
                <div className="ai">AI</div>
              </div>
              <div className="rectangle-container">
                <div className="group-item" />
                <div className="group-inner" />
                <div className="ai">채팅</div>
              </div>
              <div className="rectangle-parent1">
                <div className="group-item" />
                <div className="group-inner" />
                <div className="ai">...</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultForm;
