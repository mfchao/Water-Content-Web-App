import LiquidGlass from "liquid-glass-react";
import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

function Result() {
  const waveRef = useRef(null);
  const containerRef = useRef(null);
  const location = useLocation();
  const navigate = useNavigate();
  const prediction = location.state?.prediction;
  const image = location.state?.image;
  const [showGlass, setShowGlass] = useState(false);
  const [showRestart, setShowRestart] = useState(false);

  let waveHeight = 0;

  if (prediction < 9.3) {
    waveHeight = window.innerHeight * 0.35; // minimum height
  } else if (prediction < 15) {
    waveHeight =
      window.innerHeight * 0.15 +
      (prediction / 15) * (window.innerHeight * 0.4); // scale between 15% and 80%
  } else {
    waveHeight = window.innerHeight * 0.8; // maximum height
  }

  useEffect(() => {
    if (waveRef.current && containerRef.current) {
      var width = containerRef.current.offsetWidth;
      var containerHeight = containerRef.current.offsetHeight;
      var wave = waveRef.current;
      var waveWidth = width; // Wave SVG width (usually container width)
      var waveDelta = 20; // Wave amplitude
      var speed = 1.1; // Wave animation speed
      var wavePoints = 6; // How many point will be used to compute our wave
      var startTime = Date.now();
      var duration = 2000; // 2 seconds

      function calculateWavePoints(factor, currentHeight) {
        var points = [];

        for (var i = 0; i <= wavePoints; i++) {
          var x = (i / wavePoints) * waveWidth;
          var sinSeed = (factor + (i + (i % wavePoints))) * speed * 100;
          var sinHeight = Math.sin(sinSeed / 100) * waveDelta;
          var yPos =
            containerHeight -
            currentHeight +
            Math.sin(sinSeed / 100) * sinHeight;
          points.push({ x: x, y: yPos });
        }

        return points;
      }

      function buildPath(points) {
        var SVGString = "M " + points[0].x + " " + points[0].y;

        var cp0 = {
          x: (points[1].x - points[0].x) / 2,
          y:
            points[1].y -
            points[0].y +
            points[0].y +
            (points[1].y - points[0].y),
        };

        SVGString +=
          " C " +
          cp0.x +
          " " +
          cp0.y +
          " " +
          cp0.x +
          " " +
          cp0.y +
          " " +
          points[1].x +
          " " +
          points[1].y;

        var prevCp = cp0;
        var inverted = -1;

        for (var i = 1; i < points.length - 1; i++) {
          var cp1 = {
            x: points[i].x - prevCp.x + points[i].x,
            y: points[i].y - prevCp.y + points[i].y,
          };

          SVGString +=
            " C " +
            cp1.x +
            " " +
            cp1.y +
            " " +
            cp1.x +
            " " +
            cp1.y +
            " " +
            points[i + 1].x +
            " " +
            points[i + 1].y;
          prevCp = cp1;
          inverted = -inverted;
        }

        SVGString += " L " + width + " " + containerHeight;
        SVGString += " L 0 " + containerHeight + " Z";
        return SVGString;
      }

      function tick() {
        var now = Date.now();
        var elapsed = now - startTime;
        var currentHeight = Math.min(
          (elapsed / duration) * waveHeight,
          waveHeight
        );

        var factor = now / 1000;
        wave.setAttribute(
          "d",
          buildPath(calculateWavePoints(factor, currentHeight))
        );

        if (elapsed > duration && !showGlass) {
          setShowGlass(true);
          setShowRestart(true);
        }

        window.requestAnimationFrame(tick);
      }
      tick();
    }
  }, []);

  let waterContent = "";
  if (prediction < 9.5) {
    waterContent = "Low Water Content";
  } else if (prediction < 18) {
    waterContent = "Medium Water Content";
  } else {
    waterContent = "High Water Content";
  }

  const handleRestart = () => {
    navigate("/");
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
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <h1 style={{ fontSize: 20, fontWeight: "bold", marginTop: 80 }}>
        THIS SOIL HAS...
      </h1>
      <div
        style={{
          position: "relative",
          width: "95%",
          height: "60vh",
          borderRadius: 10,
          overflow: "hidden",
          marginTop: 20,
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
        ref={containerRef}
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          width: "100%",
          height: "100%",
        }}
      >
        <svg
          style={{
            width: "100%",
            height: "100%",
            filter: "blur(5px)",
          }}
        >
          <defs>
            <linearGradient id="waveGradient" x1="0%" y1="100%" x2="0%" y2="0%">
              <stop offset="0%" stopColor="#DAF3FF" stopOpacity="1" />
              <stop offset="80%" stopColor="#1E99D1" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#005F7F" stopOpacity="0.9" />
            </linearGradient>
          </defs>
          <path ref={waveRef} fill="url(#waveGradient)" stroke="none" />
        </svg>
        {showGlass && (
          <div
            style={{
              position: "absolute",
              bottom: waveHeight - 200,
              left: "50%",
              transform: "translateX(-50%)",
              background: "rgba(255, 255, 255, 0.2)",
              backdropFilter: "blur(10px)",
              padding: "10px 20px",
              borderRadius: "20px",
              boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)",
              textAlign: "center",
              minHeight: 50,
              maxHeight: 200,
              overflow: "hidden",
            }}
          >
            <h1
              style={{
                fontSize: 64,
                fontWeight: "bold",
                color: "black",
                padding: 0,
                margin: 0,
              }}
            >
              {prediction.toFixed(2)}
            </h1>
            <p style={{ fontSize: 13, color: "black", margin: 5 }}>
              {waterContent}
            </p>
          </div>
        )}
      </div>
      {showRestart && (
        <div
          style={{
            position: "absolute",
            bottom: 10,
            left: 0,
            width: "100%",
            textAlign: "center",
            paddingBottom: 20,
          }}
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
            onClick={handleRestart}
          >
            Restart
          </button>
        </div>
      )}
    </div>
  );
}
export default Result;
