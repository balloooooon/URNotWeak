import './Content.css'

import img1 from 'assets/images/contentThumnail01.png';
import img2 from 'assets/images/contentThumnail02.png';
import img3 from 'assets/images/contentThumnail03.png';

export const Content = () => {

    // 사진파일 가져오기 일단 기본으로
    // 썸네일 이미지, id 필요
    const imgs = [
        img1, img2, img3
    ]

    return (
        <div className="img-content scroll-bar">
            <div className='flex-box'>
            {
                // 이미지들
                imgs.map((img) => {
                    return(
                        <div className='wrapp'><img src={img}></img></div>
                    )
                }

                )
            }
                
            </div>
        </div>
    )
}

export default Content;