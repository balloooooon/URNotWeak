import React from "react";
import { Background, LoadingText } from "./LoadingStyle";
import Spinner from "../../../assets/images/spinner2.gif";

export default () => {
  return (
    <Background>
      <LoadingText>이미지 생성중 입니다.</LoadingText>
      <img src={Spinner} alt="로딩중" width="15%" />
    </Background>
  );
};
