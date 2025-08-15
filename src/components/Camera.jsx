import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import cameraIcon from "/camera.svg";

function Camera() {
  const [stream, setStream] = useState(null);
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
    navigate("/preview", { state: { image: imageData } });
  };

  if (!stream) {
    handleStartCamera();
  }

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        backgroundImage: "url(/background.png)",
        backgroundSize: "cover",
        backgroundPosition: "center",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div
        style={{
          width: "95%",
          height: "60vh",
          borderRadius: 10,
          overflow: "hidden",
          marginTop: 120,
        }}
      >
        <video
          ref={videoRef}
          autoPlay
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      </div>
      <div style={{ marginTop: 120, paddingBottom: 20 }}>
        <img
          src={cameraIcon}
          alt="Capture"
          style={{ width: 100, height: 100, cursor: "pointer" }}
          onClick={handleCapture}
        />
      </div>
    </div>
  );
}

export default Camera;
