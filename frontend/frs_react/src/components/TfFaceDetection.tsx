"use client"

import React, { useEffect, useRef, useState } from "react";
import { load, MediaPipeFaceDetectorTfjs } from "@/dnn/face_detector/detector";
import { setupBackend } from "@/dnn/tf-backend";
import { BoundingBox } from "@/dnn/shared/interfaces/shapes";
import { sendCroppedFace } from "@/api/clients/faceRecognitionClient";
import { drawDetection } from "@/dnn/shared/drawing_utils";

const cropFace = (image: HTMLVideoElement, box: BoundingBox): Promise<Blob> => {
    return new Promise((resolve) => {
        // Create a temporary canvas to crop the face
        const faceCanvas = document.createElement("canvas");
        const faceCtx = faceCanvas.getContext("2d");

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
            // Specify desired video resolution
            const constraints = {
                video: {
                    width: { ideal: 640 }, // Preferred width
                    height: { ideal: 480 }, // Preferred height
                },
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
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
        if (video.videoWidth == 0 || video.videoHeight == 0) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (!ctx) return;

        // Set canvas dimensions to match the video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Capture video frame as ImageData
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

        const predictions = await netDetectionTf.estimateFaces(video, { width: video.videoWidth, height: video.videoHeight }, { flipHorizontal: false });

        if (predictions.length > 0) {
            // console.log(predictions);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < predictions.length; i++) {
                const detection = predictions[i];

                // draw over canvas for visualization
                drawDetection(ctx, detection);

                // crop image based on bounding bbox
                const croppedFace = await cropFace(video, detection.box);
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
        <div style={{ position: "relative", width: "100%", height: "90vh" }}>
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
