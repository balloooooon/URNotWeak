import './Content.css'

export const Content = () => {

    // 사진파일 가져오기 일단 기본으로
    // 썸네일 이미지, id 필요
    const imgs = ["https://img.freepik.com/free-photo/cute-little-dog-isolated-on-yellow_23-2148985931.jpg",
"https://image.utoimage.com/preview/cp872722/2022/12/202212008462_500.jpg",
"https://img.freepik.com/free-photo/front-view-adorable-shiba-inu-dog_23-2149457807.jpg?w=360&t=st=1691718094~exp=1691718694~hmac=06fdac5e70ad01f4bac6b23f4d5de158656fd9d0c99fbea5f77cc18ddec9dfd7"
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