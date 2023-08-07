import React, {useState, useEffect} from "react";
import { Frame } from "components/Home/HomeFrame";
import { Nav } from "components/Nav/Nav";
import Chatting from "pages/Chatting/Chatting";
import './HomePage.css'

export const Home = () => {
  // 채팅 최초 1번 로딩
  useEffect(() => {
    // 두번 만들지 않도록 이미 생성되었는지 확인.
    if(window.ChannelIO === undefined) {
      Chatting.loadScript();
    }

    Chatting.boot({
      pluginKey: process.env.REACT_APP_CHAT_PLUGIN_KEY
    })
    console.log('boot')

  }, [])
  

  // 스크롤 한번으로 섹션 이동하도록
  const [currentSection, setCurrentSection] = useState(0);
  const sectionRefs = Array.from({ length: 5 }, () => React.createRef());

  useEffect(() => {
    const handleScroll = (e) => {
      const direction = e.deltaY > 0 ? 1 : -1;

      setCurrentSection((prevSection) => {
        const newSection = prevSection + direction;
        if (newSection >= 0 && newSection < sectionRefs.length) {
          sectionRefs[newSection].current.scrollIntoView({
            behavior: 'smooth',
          });
          return newSection;
        }
        return prevSection;
      });
    };

    window.addEventListener('wheel', handleScroll);
    return () => {
      window.removeEventListener('wheel', handleScroll);
    };
  }, [sectionRefs]);


  return (
    <div className="display">
      <Nav className="nav-instance" />
      <div className="home">
        <div className="homepages">
          <div ref={sectionRefs[0]} className={`section ${currentSection === 0 ? 'active' : ''}`}>
            <Frame
              link=""
              backimage="home1"
              text1="대한민국의 마약 근절을 위해 노력합니다."
              text2=""
            />
          </div>
          <div ref={sectionRefs[1]} className={`section ${currentSection === 0 ? 'active' : ''}`}>
            <Frame
              link="simul"
              backimage="home2"
              text1="‘한번쯤은’ 이라는 생각, 두 번 다시 돌아오지 못합니다."
              text2="시뮬레이션 확인하기"
            />
          </div>
          <div ref={sectionRefs[2]} className={`section ${currentSection === 0 ? 'active' : ''}`}>
            <Frame
              link="test"
              backimage="home3"
              text1="누구든 시작할 수 있습니다. 하지만 벗어날 수 없습니다."
              text2="마약 취약성 확인하기"
            />
          </div>
          <div ref={sectionRefs[3]} className={`section ${currentSection === 0 ? 'active' : ''}`}>
            <Frame
              link="ai/upload"
              backimage="home4"
              text1="망가져 가는 내모습 유지할 수 없는 일상"
              text2="얼굴AI"
            />
          </div>
          <div ref={sectionRefs[4]} className={`section ${currentSection === 0 ? 'active' : ''}`}>
            <Frame
              link="chatting"
              backimage="home5"
              text1="궁금한 내용이 더 생겼나요? 도움이 필요하신가요?"
              text2="chatting"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
