import React from "react";
import { Nav } from "components/Nav/Nav";
import { HomeForm } from "components/Home/Home"

export const Home = () => {
    return (
        <div>
            <Nav className="nav-instance" />
            <HomeForm />
        </div>
    );
};

export default Home;