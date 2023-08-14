import { Link } from "react-router-dom";
import './Select.css'

export const Select = () => {

    return(
        <div className="select">
            <p className="tx-t">필터를 선택하여 확인할 수 있어요</p>
            <div className="wrapper"><Link to={`filter1`} className="tx-t" style={{ textDecoration: "none" }}>filter1</Link></div>
            <div className="wrapper"><Link to={`filter2`} className="tx-t" style={{ textDecoration: "none" }}>filter2</Link></div>
            <div className="wrapper"><Link to={`filter3`} className="tx-t" style={{ textDecoration: "none" }}>filter3</Link></div>
            <div className="wrapper"><Link to={`filter4`} className="tx-t" style={{ textDecoration: "none" }}>filter4</Link></div>
            <div className="wrapper"><Link to={`filter5`} className="tx-t" style={{ textDecoration: "none" }}>filter5</Link></div>
        </div>
    )
}

export default Select;