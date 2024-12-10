"use client"

import React, { useEffect, useRef, useState } from "react";
import { load, MediaPipeFaceDetectorTfjs } from "@/dnn/face_detector/detector";
import { setupBackend } from "@/dnn/tf-backend";
import { BoundingBox } from "@/dnn/shared/interfaces/shapes";
import { sendCroppedFace } from "@/api/clients/faceRecognitionClient";

const cropFace = (image: HTMLVideoElement, box: BoundingBox) => {
    return new Promise((resolve) => {
        // Create a temporary canvas to crop the face
        const faceCanvas = document.createElement("canvas");
        const faceCtx = faceCanvas.getContext("2d");

        // faceCanvas.width = Math.abs(box.width);
        // faceCanvas.height = Math.abs(box.height);

        faceCanvas.width = Math.abs(image.videoWidth);
        faceCanvas.height = Math.abs(image.videoHeight);

        // Draw the cropped face onto the temporary canvas
        faceCtx?.drawImage(
            image, // The main canvas containing the video
            0,
            // Math.max(Math.abs(box.xMin) - box.width / 2, 0),
            // Math.max(Math.abs(box.yMin) - box.height / 2, 0),
            0,
            // Math.abs(box.width),
            image.videoWidth,
            // Math.abs(box.height),
            image.videoHeight,
            0,
            0,
            faceCanvas.width,
            faceCanvas.height
        );

        // return faceCanvas.toDataURL("image/jpeg"); // Convert the cropped face to a data URL
        return faceCanvas.toBlob((blob) => resolve(blob!), "image/jpeg");
    });
};



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

        // Capture video frame as ImageData
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

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
                ctx.fillStyle = 'rgba(255, 0, 0, 0.2)';
                ctx.fillRect(Math.abs(box.xMin), Math.abs(box.yMin), size[0], size[1]);

                // Draw rectangle outline
                ctx.strokeStyle = 'rgba(100, 149, 237, 0.8)'; // Set stroke color and transparency
                ctx.lineWidth = 2; // Set the line width for better visibility
                ctx.strokeRect(
                    Math.abs(box.xMin),
                    Math.abs(box.yMin),
                    size[0],
                    size[1]
                );

                const landmarks = prediction.keypoints;

                ctx.fillStyle = 'blue';
                for (let j = 0; j < landmarks.length; j++) {
                    const x = Math.abs(landmarks[j].x);
                    const y = Math.abs(landmarks[j].y);
                    ctx.fillRect(x, y, 5, 5);
                }

                // todo: crop image based on bbox
                const croppedFace = await cropFace(video, box);
                // todo: send cropped image python backend
                const response = await sendCroppedFace(croppedFace);
                console.log(response);
            }
        }
    };


    useEffect(() => {
        const interval = setInterval(detectFrame, 100); // Detect faces every 100ms
        return () => clearInterval(interval);
    }, [netDetectionTf]);

    return (
        <div style={{ position: "relative", width: "100%", height: "80vh" }}>
            <video
                ref={videoRef}
                style={{
                    display: "block",
                    width: "auto", height: "100%"
                }}
            />
            <canvas
                ref={canvasRef}
                style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "auto",
                    height: "100%",
                }}
            />
        </div>
    );
};

export default TfFaceDetection;
