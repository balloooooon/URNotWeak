import twitter from "../../../assets/images/트위터.png";
import kakaoimg from "../../../assets/images/kakao.png";
import link from "../../../assets/images/share.png";

export const Share = () => {
    const handleKakaoShare = () => {
        if (window.Kakao) {
          window.Kakao.Link.sendDefault({
            objectType: "feed",
            content: {
              title: "친구에게 공유하기",
              description: "약해지지마 - 시뮬레이션을 체험하시겠습니까?",
              imageUrl: kakaoimg,
              link: {
                mobileWebUrl: "https://urnotweak.site",
                webUrl: "https://urnotweak.site",
              },
            },
            buttons: [
              {
                title: "웹으로 보기",
                link: {
                  mobileWebUrl: "https://urnotweak.site",
                  webUrl: "https://urnotweak.site",
                },
              },
            ],
          });
        }
      };
    
    const shareToTwitter = (url) => {
    const encodedUrl = encodeURIComponent(url);
    console.log("Encoded URL:", encodedUrl);
    // const twitterUrl = `https://twitter.com/intent/tweet?url=${encodedUrl}`;
    const twitterUrl = `https://twitter.com/intent/tweet?url=https://urnotweak.site/ 약해지지마 시뮬레이션 체험해보기 !`;
    window.open(twitterUrl, "_blank");
    };

    const urlToShare = "https://urnotweak.site/";
    console.log(urlToShare);
    
    const copyToClipboard = (text) => {
        const el = document.createElement("textarea");
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand("copy");
        document.body.removeChild(el);
        window.alert("링크가 클립보드에 복사되었습니다.");
    };

    return(
        <div className="result-item mt">
            <div className="line"></div>
            <p className="tx-s">친구에게 공유하기</p>
            <div className="btn-group">
                <img
                className="circle"
                alt="카톡공유이미지"
                src={kakaoimg}
                onClick={handleKakaoShare}
                />
                <img
                className="circle"
                alt="트위터공유"
                src={twitter}
                onClick={() => shareToTwitter(urlToShare)}
                />
                <div className="circle" >
                <img
                className="link"
                alt="링크공유"
                src={link}
                onClick={() => copyToClipboard(urlToShare)}
                /></div>
            </div>
        </div>
    )
}

export default Share;