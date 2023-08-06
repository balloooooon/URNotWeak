import React, {useState, useEffect} from "react";
import {useNavigate} from "react-router-dom";

export const Content = ({list}) => {
  const [index, setIndex] = useState(1);
  const [img, setImg] = useState();
  const [ans0, setAns0] = useState({});
  const [ans1, setAns1] = useState({});
  const [score, setScore] = useState(0);

  const navigate = useNavigate();

  const goUpload = () => {
    navigate('/ai/upload');
  }

  useEffect(() => {
    console.log(index+'contenteffect');
    console.log(list);
    for(let i = 0 ; i < list.length ; i++) {
      if(list[i].testQNo == index) {
        setImg(list[i].testQImg)
        setAns0(list[i].testAnswers[0]);
        setAns1(list[i].testAnswers[1]);
        break;
      }
    }
  }, [index]);

  const btn = (btn) => {
    if(btn == 0) setScore((pre)=> pre+ans0.testAScore);
    else setScore((pre)=> pre+ans1.testAScore);
  }

  
  const btn0 = () => {
    setScore((pre)=> pre+ans0.testAScore);
    next();
  }

  const btn1 = () => {
    setScore((pre)=> pre+ans1.testAScore);
    next();
  }


  const next = () => {
    if(index == list.length) navigate({
      pathname: '/test/result',
      search: `score=${score}`
    });
    else setIndex((pre)=> pre+1);
  }

  return (
    <div className="upload">
      <img 
        className="uploadimg"
        src={img} 
        />
      <div className="btn" onClick={btn0}>{ans0.testAContent}</div>
      <div className="btn" onClick={btn1}>{ans1.testAContent}</div>
      <div className="btn" onClick={goUpload}>다시해보기</div>
    </div>
  );
};
    
export default Content;
