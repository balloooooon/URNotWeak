import React , { useState, useEffect } from "react";
import { Background, LoadingText } from "./LoadingStyle";
import Spinner from "../../../assets/images/spinner2.gif";
import axios from "axios";

export default () => {
  const [imageSrc, setImageSrc] = useState(null);

  axios({
    method: "GET",
    url: "https://urnotweak.site:8589/content/img",
  })
  .then((response) => {
    const img = response.data;
    setImageSrc(img);
  })
  .catch((Error) => {
    console.log(Error);
  });

  return (
    <Background>
      <img className="loadingImg" src={imageSrc} width="80%"/>
      <img src={Spinner} alt="로딩중" width="15%" />
    </Background>
  );
};
