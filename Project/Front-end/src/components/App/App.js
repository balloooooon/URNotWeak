import React, {lazy, Suspense} from "react";
import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";


// // import Home from "pages/Homepage/HomePage"
// import Simulation from "pages/Simulation/Simulation"
// import AI from 'pages/AI/AI'
// import AR from 'pages/AR/AR'
// import Test from 'pages/Test/Test'
// import NotFound from 'pages/NotFound/NotFound'


// import Content from 'pages/Content/Content'
import 'fonts/font.css'
import Loading from 'pages/Loading/Loading';

const Simulation = lazy(() => import('pages/Simulation/Simulation'));
const AI = lazy(() => import('pages/AI/AI'));
const AR = lazy(() => import('pages/AR/AR'));
const Test = lazy(() => import('pages/Test/Test'));
const Content = lazy(() => import('pages/Content/Content'));
const Home = lazy(() => import('pages/Homepage/HomePage'));
const NotFound = lazy(() => import('pages/NotFound/NotFound'));

function App() {
  return (
    <div className="display">
      <Router>
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="simul" element={<Simulation />} />
            <Route path="ai/*" element={<AI />} />
            <Route path="ar/*" element={<AR />} />
            <Route path="test/*" element={<Test />} />
            <Route path="content/*" element={<Content />} />
            <Route path="/*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </Router>
    </div>
  );
}

export default App;