"use client"

import React, { useEffect, useRef, useState } from "react";
import { load, MediaPipeFaceDetectorTfjs } from "@/dnn/face_detector/detector";
import { setupBackend } from "@/dnn/tf-backend";


const TfFaceDetection = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [netDetectionTf, setNetDetectionTf] = useState<MediaPipeFaceDetectorTfjs | null>(null);

    useEffect(() => {
        // Load TensorFlow model
        const loadModel = async () => {
            await setupBackend();

            const model = await load();
            setNetDetectionTf(model);
        };
        loadModel();

        // Initialize video stream
        const startVideo = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play();
                };
            }
        };
        startVideo();

        return () => {
            if (videoRef.current && videoRef.current.srcObject) {
                (videoRef.current.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
            }
        };
    }, []);

    const detectFrame = async () => {
        if (!videoRef.current || !canvasRef.current || !netDetectionTf) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (!ctx) return;

        // Set canvas dimensions to match the video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        console.log(`w: ${video.videoWidth} h: ${video.videoHeight}`);

        // Capture video frame as ImageData
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

        const annotateBoxes = true;
        const predictions = await netDetectionTf.estimateFaces(video, { width: video.videoWidth, height: video.videoHeight }, { flipHorizontal: false });
        console.log(`predictions length: ${predictions.length}`);

        if (predictions.length > 0) {
            console.log(predictions);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < predictions.length; i++) {
                const prediction = predictions[i];
                const box = prediction.box;

                const size = [Math.abs(box.width), Math.abs(box.height)];
                console.log(`size: ${size}`);
                ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
                ctx.fillRect(Math.abs(box.xMin), Math.abs(box.yMin), size[0], size[1]);

                // Draw rectangle outline
                ctx.strokeStyle = 'rgba(255, 0, 0, 0.8)'; // Set stroke color and transparency
                ctx.lineWidth = 2; // Set the line width for better visibility
                ctx.strokeRect(
                    Math.abs(box.xMin),
                    Math.abs(box.yMin),
                    size[0],
                    size[1]
                );

                if (annotateBoxes) {
                    const landmarks = prediction.keypoints;

                    ctx.fillStyle = 'blue';
                    for (let j = 0; j < landmarks.length; j++) {
                        const x = Math.abs(landmarks[j].x);
                        const y = Math.abs(landmarks[j].y);
                        ctx.fillRect(x, y, 5, 5);
                    }
                }
            }
        }
        // requestAnimationFrame(detectFrame);
    };


    useEffect(() => {
        const interval = setInterval(detectFrame, 100); // Detect faces every 100ms
        return () => clearInterval(interval);
    }, [netDetectionTf]);

    return (
        <div style={{ position: "relative" }}>
            <video
                ref={videoRef}
                style={{
                    display: "block",
                    width: 640, height: 480
                }}
            />
            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: 640,
                    height: 480,
                }}
            />
        </div>
    );
};

export default TfFaceDetection;
