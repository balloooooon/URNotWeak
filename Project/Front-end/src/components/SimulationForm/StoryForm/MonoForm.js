import "./MonoForm.css";
import NextBtn from "../NextBtn";
const MonoForm = () => {
  return (
    <div className="mono">
      <NextBtn />
      <div className="mono-wrapper">
        <div className="mono-txt">
          <p className="p">너무 힘든데...</p>
          <p className="p">&nbsp;</p>
          <p className="p">딱 한 번만 더 해볼까?</p>
        </div>
      </div>
    </div>
  );
};

export default MonoForm;
