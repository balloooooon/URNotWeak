import React , { useState, useEffect } from "react";
import { Background, LoadingText } from "./LoadingStyle";
import Spinner from "../../../assets/images/loading2.gif";
import axios from "axios";

export default (props) => {
  const [imageSrc, setImageSrc] = useState(null);

  console.log(props);
  useEffect(() => {
    console.log(props);
    setImageSrc(props.loadingImg || null);
  }, [props]);

  return (
    <Background>
      <img className="loadingImg" src={imageSrc} width="80%"/>
      <img src={Spinner} alt="로딩중" width="15%" />
    </Background>
  );
};
