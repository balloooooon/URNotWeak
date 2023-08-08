import React, {useState, useEffect} from "react";
import { Link } from "react-router-dom";
import { Menu } from './Menu'
import "./Nav.css";
import SideMenu from 'pages/SideMenu/Menu';

export const Nav = () => {
  //sidebar menu를 위한 state
  const [isMenu,setMenu] = useState(false)

  // useEffect(() => {
  //   console.log(isMenu);
  // })

  // 바로 setMenu를 이용하면 무한루프렌더링이 걸린다.
  // 함수를 거쳐가도록 만듬.
  function changeMenu() {
    setMenu(!isMenu);
  }

  return (
    <div className="nav">
      <div className="navbar">
        <Link to="/" className="element" style={{ textDecoration: "none" }}>약해지지마</Link>
        <div className="menu-screen" onClick={changeMenu}>
          <Menu className="m-menu" color="white"></Menu>
        </div>
      </div>
      {isMenu ?
        <SideMenu closebtn={changeMenu}/>: <></>
      }
    </div>
  );
};

export default Nav;
