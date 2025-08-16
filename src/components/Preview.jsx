import React from "react";
import { useNavigate, useLocation } from "react-router-dom";

function Preview() {
  const navigate = useNavigate();
  const location = useLocation();
  const image = location.state?.image;

  const handleRetake = () => {
    navigate("/camera");
  };

  function getApiUrl() {
    if (window.location.hostname === "localhost") {
      return "http://127.0.0.1:5000";
    } else {
      return ""; // or "/api" if you want to be explicit
    }
  }

  const handlePredict = async () => {
    try {
      const formData = new FormData();
      const blob = await fetch(image).then((res) => res.blob());
      const file = new File([blob], "image.jpg", { type: "image/jpeg" });
      formData.append("image", file);
      const apiUrl = getApiUrl();

      const response = await fetch(`${apiUrl}/api/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        console.error(data.error);
      } else {
        navigate("/result", {
          state: { prediction: data.prediction, image: image },
        });
      }
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        display: "flex",
        backgroundImage: "url(/background.png)",
        backgroundSize: "cover",
        backgroundPosition: "center",
        flexDirection: "column",
        justifyContent: "space-between",
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
        <img
          src={image}
          alt="Captured image"
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      </div>
      <div
        style={{ paddingBottom: 120, display: "flex", alignItems: "center" }}
      >
        <button
          style={{
            all: "unset",
            border: "none",
            cursor: "pointer",
            fontFamily: "Arial Narrow",
            fontWeight: 600,
            fontSize: 15,
            color: "black",
            background: "rgba(255, 255, 255, 0.2)",
            backdropFilter: "blur(10px)",
            padding: "10px 15px",
            borderRadius: "20px",
            boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)",
          }}
          onClick={handleRetake}
        >
          RETAKE
        </button>
        <span
          style={{
            margin: "0 40px",
            color: "black",
            fontWeight: 600,
            fontSize: 13,
          }}
        >
          or
        </span>
        <button
          style={{
            all: "unset",
            border: "none",
            cursor: "pointer",
            fontFamily: "Arial Narrow",
            fontWeight: 600,
            fontSize: 15,
            color: "black",
            background: "rgba(255, 255, 255, 0.2)",
            backdropFilter: "blur(10px)",
            padding: "10px 15px",
            borderRadius: "20px",
            boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)",
          }}
          onClick={handlePredict}
        >
          PREDICT
        </button>
      </div>
    </div>
  );
}
export default Preview;
