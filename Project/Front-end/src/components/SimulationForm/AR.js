import NextBtn from "./NextBtn";
import "./AR.css"

const AR = ({ onNext }) => {
  return (
    <>
      <NextBtn onClick={onNext} />
      <div className="p123">

      여기는 사라질 페이지 입니다
      </div>
      <p></p>
      <p></p>
      <p></p>
      <p></p>
      <p></p>
      <p>AR</p>
    </>
  );
};

export default AR;
