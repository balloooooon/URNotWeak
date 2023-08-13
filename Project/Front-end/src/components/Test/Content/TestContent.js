import React, {useState, useEffect} from "react";
import {useNavigate} from "react-router-dom";
import './TestContent.css'

export const Content = ({list}) => {
  const [index, setIndex] = useState(1);
  const [img, setImg] = useState();
  const [ans0, setAns0] = useState({});
  const [ans1, setAns1] = useState({});
  const [score, setScore] = useState(0);

  const navigate = useNavigate();

  // index가 바뀔때 사진이랑 문항 새로고침
  useEffect(() => {
    if(list === undefined) {  // 혹시 새로고침 했으면 list가 비어있으니까 뒤로가기
      navigate(-1);
    }
    else {
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

    }
  }, [index]);

  // 현재 컴포넌트에 새로고침 이벤트 달기
  useEffect(() => {
    (()=> {
      window.addEventListener("beforeunload", refreshListener);
    })();

    return () => {
      window.removeEventListener("beforeunload", refreshListener);
    };
    
  },[]) 

  //새로고침 이벤트가 발생하면 새로고침/취소에 상관없이 실행
  const refreshListener = (e) => {
    e.preventDefault(); //기본클릭 동작방지
    e.returnValue = "";
  }

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
      <div className="btn test-btn tx-t" onClick={btn0}>{ans0.testAContent}</div>
      <div className="btn test-btn tx-t" onClick={btn1}>{ans1.testAContent}</div>
    </div>
  );
};
    
export default Content;
