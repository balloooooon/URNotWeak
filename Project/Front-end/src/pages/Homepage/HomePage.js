import React from "react";
import { Frame } from "components/Home/HomeFrame";
import { Nav } from "components/Nav/Nav";
import Chatting from "pages/Chatting/Chatting";
import './HomePage.css'

export const Home = () => {

  Chatting.loadScript();

  Chatting.boot({
    pluginKey: "0c4aec1b-7fe4-4693-bafb-49862c83f43e",
  })

  return (
    <div className="display">
      <Nav className="nav-instance" />
      <div className="home">
        <div className="homepages">
          <Frame
            link=""
            backimage="home1"
            text1="대한민국의 마약 근절을 위해 노력합니다."
            text2=""
          />
          <Frame
            link="simul"
            backimage="home2"
            text1="‘한번쯤은’ 이라는 생각, 두 번 다시 돌아오지 못합니다."
            text2="시뮬레이션 확인하기"
          />
          <Frame
            link="test"
            backimage="home3"
            text1="누구든 시작할 수 있습니다. 하지만 벗어날 수 없습니다."
            text2="마약 취약성 확인하기"
          />
          <Frame
            link="ai/upload"
            backimage="home4"
            text1="망가져 가는 내모습 유지할 수 없는 일상"
            text2="얼굴AI"
          />
          <Frame
            link="chatting"
            backimage="home5"
            text1="궁금한 내용이 더 생겼나요? 도움이 필요하신가요?"
            text2="chatting"
          />
        </div>
      </div>
    </div>
  );
};

export default Home;
