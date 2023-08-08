import "./MonoForm.css";
import NextBtn from "../NextBtn";
const MonoForm = ({onNext,text}) => {
  return (
    <div className="mono">
      <NextBtn onClick={onNext}/>
      <div className="mono-wrapper">
        <div className="mono-txt">
          <p className="p">{text}</p>
        </div>
      </div>
    </div>
  );
};

export default MonoForm;
