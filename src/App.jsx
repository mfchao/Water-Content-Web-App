import React, { useState, useEffect, useRef } from "react";
import {
  BrowserRouter,
  Route,
  Routes,
  useNavigate,
  useLocation,
} from "react-router-dom";

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

function Home() {
  const navigate = useNavigate();

  return (
    <div>
      <button onClick={() => navigate("/camera")}>Take a Photo</button>
    </div>
  );
}

function Camera() {
  const [stream, setStream] = useState(null);
  const [image, setImage] = useState(null);
  const navigate = useNavigate();
  const videoRef = useRef(null);

  const handleStartCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    setStream(stream);
  };

  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  const handleCapture = () => {
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL();
    setImage(imageData);
    navigate("/preview", { state: { image: imageData } });
  };

  if (!stream) {
    handleStartCamera();
  }

  return (
    <div>
      <video ref={videoRef} autoPlay />
      <button onClick={handleCapture}>Capture</button>
    </div>
  );
}

function Preview() {
  const navigate = useNavigate();
  const location = useLocation();
  const image = location.state?.image;

  const handleRetake = () => {
    navigate("/camera");
  };

  const handlePredict = async () => {
    try {
      const formData = new FormData();
      const blob = await fetch(image).then((res) => res.blob());
      const file = new File([blob], "image.jpg", { type: "image/jpeg" });
      formData.append("image", file);
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        console.error(data.error);
      } else {
        navigate("/result", { state: { prediction: data.prediction } });
      }
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <img src={image} alt="Captured image" />
      <button onClick={handleRetake}>Retake</button>
      <button onClick={handlePredict}>Predict</button>
    </div>
  );
}

function Result() {
  const navigate = useNavigate();
  const location = useLocation();
  const prediction = location.state.prediction;

  const handleRestart = () => {
    navigate("/", { replace: true }); // Navigate to the home page and replace the current entry in the history stack
  };

  return (
    <div>
      <p>Prediction: {prediction}</p>
      <button onClick={handleRestart}>Restart</button>
    </div>
  );
}

export default App;
