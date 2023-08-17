import React, {useState, useEffect, useRef} from "react";
import { Link } from "react-router-dom";
import { Menu } from './Menu'
import "./Nav.css";
import SideMenu from 'pages/SideMenu/Menu';

export const Nav = ({width=270}) => {
  //sidebar menu를 위한 state
  const [css, setCss] = useState("back-default");
  const [isOpen, _setOpen] = useState(false);
  const [xPosition, setX] = useState(-width);
  const side = useRef();

  // button 클릭 시 토글
  const toggleMenu = (e) => {
    if (isOpen) { // 열림 -> 닫힘
      setX(-width);
      setCss("back-default");
      console.log('닫기')
    } else {
      setX(0);
      setCss("back-white");
      console.log('열기')
    }
    _setOpen(pre => !pre);
    // 이벤트 버블링을 막기위함.
    if(e!= undefined) e.stopPropagation();
  };
  
  // 사이드바 외부 클릭시 닫히는 함수
  const handleClose = e => {
    let sideArea = side.current;
    if (sideArea) {
      let sideChildren = sideArea.contains(e.target);
      if (isOpen && !sideChildren) {
        console.log('외부클릭');
        toggleMenu();
      }
    }
  }

  // 여기 개념 정리하기!
  // 이벤트 캡쳐링, 버블링
  useEffect(()=> {
    window.addEventListener('click', handleClose);  //클로저
    return () => {
      window.removeEventListener('click', handleClose);
    };
  }, [isOpen])

  return (
    <div className="nav">
      <div className={`navbar ${css}`}>
        <Link to="/" className="element tx-rt" style={{ textDecoration: "none" }}>URNotWeak</Link>
        <div className="menu-screen" onClick={(e)=>toggleMenu(e)}>
          <Menu className="m-menu"></Menu>
        </div>
      </div>

      {isOpen? <div className={`non-background ${css}`}></div>: <></>}
      <div ref={side}  className="sidebar" style={{ width: `${width}px`, transform: `translatex(${-xPosition}px)`}}>
        <SideMenu closebtn={(e)=>toggleMenu(e)}/>
      </div>
      
    </div>
  );
};

export default Nav;