import React, { useRef } from "react";
import { useNavigate } from "react-router-dom";
import LiquidGlass from "liquid-glass-react";
import waterDrop from "/water-drop.svg";

function Home() {
  const navigate = useNavigate();

  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = () => {
      navigate("/preview", { state: { image: reader.result } });
    };
    reader.readAsDataURL(file);
  };

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
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <div
        style={{
          flex: 1,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <img
            src={waterDrop}
            alt="Water Drop"
            style={{ width: 20, height: 20, marginRight: 10 }}
          />
          <h1 style={{ fontSize: 20, fontWeight: "bold", color: "white" }}>
            WATER CONTENT AI
          </h1>
        </div>
      </div>
      <div style={{ paddingBottom: 100, textAlign: "center" }}>
        <p
          style={{
            color: "black",
            fontWeight: 600,
            fontSize: 13,
            marginBottom: 50,
          }}
        >
          To get started, first...
        </p>
        <div>
          <button
            style={{
              all: "unset",
              cursor: "pointer",
              fontFamily: "Arial Narrow",
              fontWeight: 600,
              fontSize: 15,
              color: "black",
              background: "rgba(255, 255, 255, 0.015)",
              backdropFilter: "blur(20px)",
              padding: "10px 15px",
              borderRadius: "20px",
              border: "1px solid rgba(255,255,255,0.2)",
              boxShadow: "0 0 10px rgba(0, 0, 0, 0.2)",
              marginRight: 20,
            }}
            onClick={() => navigate("/camera")}
          >
            TAKE A PHOTO
          </button>

          <span
            style={{
              marginRight: 20,
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
              cursor: "pointer",
              fontFamily: "Arial Narrow",
              fontWeight: 600,
              fontSize: 15,
              color: "black",
              background: "rgba(255, 255, 255, 0.015)",
              backdropFilter: "blur(20px)",
              padding: "10px 15px",
              borderRadius: "20px",
              border: "1px solid rgba(255,255,255,0.2)",
              boxShadow: "0 0 10px rgba(0, 0, 0, 0.2)",
            }}
            onClick={() => fileInputRef.current.click()}
          >
            UPLOAD A FILE
          </button>
          <input
            ref={fileInputRef}
            type="file"
            style={{ display: "none" }}
            onChange={handleFileSelect}
            accept="image/*" // Optional: restrict file types to images
          />
        </div>
      </div>
    </div>
  );
}

export default Home;
