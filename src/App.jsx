import { BrowserRouter, Route, Routes } from "react-router-dom";
import LiquidGlass from "liquid-glass-react";
import Home from "./components/Home";
import Camera from "./components/Camera";
import Preview from "./components/Preview";
import Result from "./components/Result";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/camera" element={<Camera />} />
        <Route path="/preview" element={<Preview />} />
        <Route path="/result" element={<Result />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
