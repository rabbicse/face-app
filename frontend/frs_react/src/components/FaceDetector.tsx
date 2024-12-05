"use client";

import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

const FaceDetectionTfjs = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [model, setModel] = useState<tf.GraphModel | null>(null);

    useEffect(() => {
        const loadModel = async () => {
            try {
                // Load a pre-trained TensorFlow.js face detection model (example: BlazeFace)
                const loadedModel = await tf.loadGraphModel(
                    "/models/face-detection/model.json",
                );
                setModel(loadedModel);
                console.log("Model loaded successfully");
            } catch (error) {
                console.error("Error loading the TensorFlow.js model:", error);
            }
        };

        loadModel();

        // Access the webcam
        if (navigator.mediaDevices?.getUserMedia) {
            navigator.mediaDevices
                .getUserMedia({ video: true })
                .then((stream) => {
                    if (videoRef.current) {
                        videoRef.current.srcObject = stream;
                        videoRef.current.play();
                    }
                })
                .catch((error) => console.error("Error accessing webcam:", error));
        }
    }, []);

    const detectFaces = async () => {
        if (!model || !videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (!ctx) return;

        // Set canvas dimensions to match the video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Capture the current video frame
        const inputTensor = tf.tidy(() => {
            const videoFrame = tf.browser.fromPixels(video); // Create a tensor from video frame
            const resizedFrame = tf.image.resizeBilinear(videoFrame, [192, 192]); // Resize for model
            const normalizedFrame = resizedFrame.div(255.0); // Normalize pixel values
            return normalizedFrame.expandDims(0); // Add batch dimension
        });

        try {
            // Perform face detection
            console.info("performing detection...");
            const predictions = await model.executeAsync(inputTensor);            

            // Process the predictions (example based on BlazeFace output)
            const [boxes, scores] = predictions as tf.Tensor[];

            console.info(`predictions: ${boxes} scores: ${scores}`);

            // Clear canvas for new drawings
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const threshold = 0.5; // Confidence threshold
            const boxesData = await boxes.array();
            const scoresData = await scores.array();

            boxesData.forEach((box: number[], index: number) => {
                if (scoresData[index] > threshold) {
                    const [yMin, xMin, yMax, xMax] = box;

                    // Convert normalized box coordinates to pixel values
                    const startX = xMin * canvas.width;
                    const startY = yMin * canvas.height;
                    const boxWidth = (xMax - xMin) * canvas.width;
                    const boxHeight = (yMax - yMin) * canvas.height;

                    // Draw bounding box
                    ctx.strokeStyle = "red";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(startX, startY, boxWidth, boxHeight);
                }
            });

            // Dispose tensors
            tf.dispose([boxes, scores]);
        } catch (error) {
            console.error("Error detecting faces:", error);
        } finally {
            tf.dispose(inputTensor); // Clean up input tensor
        }
    };

    useEffect(() => {
        const interval = setInterval(detectFaces, 100); // Detect faces every 100ms
        return () => clearInterval(interval);
    }, [model]);

    return (
        <div style={{ position: "relative" }}>
            <video
                ref={videoRef}
                style={{ display: "block", width: "100%" }}
            />
            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    height: "100%",
                }}
            />
        </div>
    );
};

export default FaceDetectionTfjs;
