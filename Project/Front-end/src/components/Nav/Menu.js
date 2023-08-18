export const Menu = ({ className }) => {
    return (
        <svg
            className={`menu ${className}`}
            fill="none"
            height="24"
            viewBox="0 0 24 24"
            width="24"
            xmlns="http://www.w3.org/2000/svg"
        >
            <g className="g" clipPath="url(#normalized_0)">
                <path className="path" d="M3 18H21V16H3V18ZM3 13H21V11H3V13ZM3 6V8H21V6H3Z" fill="white" />
            </g>
            <defs className="defs">
                <clipPath className="clip_path" id="normalized_0">
                    <rect className="rect" fill="white" height="24" width="24" />
                </clipPath>
            </defs>
        </svg>
    );
};