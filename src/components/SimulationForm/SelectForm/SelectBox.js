import React, { useEffect, useState } from "react";
import axios from "axios";
import "./SelectBox.css";

const SelectBox = ({ ThumbClick }) => {
  const [thumbnails, setThumbnails] = useState([]);

  useEffect(() => {
    axios
      .get("https://www.urnotweak.site:8589/simulation/thumbnail")
      .then((response) => {
        setThumbnails(response.data);
      })
      .catch((error) => {
        console.error("Error fetching thumbnails:", error);
      });
  }, []);

  const handleThumbClick = (index) => {
    ThumbClick(index);
  };

  return (
    <div className="box">
      {thumbnails.map((thumbnail, index) => (
        <div key={index} className={`ImgBox${index + 1}`}>
          <video
            className={`Thumnail${index + 1}`}
            alt=""
            src={thumbnail.url}
            onClick={() => handleThumbClick(index + 1)}
            autoPlay
            loop
            style={{ objectFit: "cover" }}
          />
        </div>
      ))}
    </div>
  );
};

export default SelectBox;
