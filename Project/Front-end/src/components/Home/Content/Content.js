import './Content.css'

export const Content = () => {

    // 사진파일 가져오기 일단 기본으로
    // 썸네일 이미지, id 필요
    const imgs = [
        "contentThumnail01","contentThumnail02","contentThumnail03"
    ]

    return (
        <div className="img-content scroll-bar">
            <div className='flex-box'>
            {
                // 이미지들
                imgs.map((img) => {
                    return( 
                        <picture className="wrapp" >
                            <source srcSet={require(`assets/images/${img}.webp`)} type="image/webp" />
                            <img src={require(`assets/images/${img}.jpg`)} alt='content-thumnail-img'></img>
                        </picture>  
                    )
                }

                )
            }
                
            </div>
        </div>
    )
}

export default Content;