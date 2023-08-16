import React, {useState, useEffect} from "react";
import {useNavigate} from "react-router-dom";
import './TestContent.css'
import axios from "axios";

export const Content = ({list}) => {
  const [index, setIndex] = useState(1);
  const [img, setImg] = useState();
  const [ans0, setAns0] = useState({});
  const [ans1, setAns1] = useState({});
  const [score, setScore] = useState(0);

  const navigate = useNavigate();

  // index가 바뀔때 사진이랑 문항 새로고침
  useEffect(() => {
    if(list === undefined || list.length === undefined) {  // 혹시 새로고침 했으면 list가 비어있으니까 뒤로가기
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

  // 마지막 결과 저장.
  // result랜더링될때 저장하면 결과 아님에도 올라가니까
  // 결과가 나왔을 때 저장하고 페이지 이동.
  const saveResult = async () => {
    // 현재 사용자의 번호 가져오기
    const user = await axios
    .get(process.env.REACT_APP_SERVER_API_URL + "/user/userno")
    .catch((Error) => {
      console.log(Error);
    });
    let userno = user.data + 1;

    // 사용자 점수 저장
    await axios({
    method: "POST",
    url: process.env.REACT_APP_SERVER_API_URL + "/user/result",
    headers: {
      "Content-Type": "application/json",
    },
    data: {
      user_id: userno,
      test_final_score: score,
    },
    })
    .then((response) => {
      console.log(response);
    })
    .catch((Error) => {
      console.log(Error);
    });

  }


  const next = async () => {
    if(index == list.length) {
        saveResult();
    
        navigate({
        pathname: '/test/result',
        search: `score=${score}`
      });
    }
    else setIndex((pre)=> pre+1);
  }

  return (
    <div className="test">
      <p className="test-main tx-b">취약성 테스트</p>
      <div className="progress">
        {list?<a className="tx-s">{index}/{list.length}</a>:null}
      </div>
      <img 
        className="testimg"
        src={img} 
        />
      <div className="btn test-btn tx-t" onClick={btn0}>{ans0.testAContent}</div>
      <div className="btn test-btn tx-t" onClick={btn1}>{ans1.testAContent}</div>
    </div>
  );
};
    
export default Content;
