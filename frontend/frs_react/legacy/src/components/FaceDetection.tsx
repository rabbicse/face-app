"use client"

import React, { useRef, useEffect, useState } from 'react';
import '@mediapipe/face_detection';
import '@tensorflow/tfjs-core';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgpu';
import * as faceDetection from '@tensorflow-models/face-detection';



import * as tf from '@tensorflow/tfjs';
// // Register WebGL backend.
// import '@tensorflow/tfjs-backend-webgl';
// import * as faceDetection from '@tensorflow-models/face-detection';
// import { MediaPipeFaceDetectorTfjsModelConfig } from '@tensorflow-models/face-detection';

const FaceDetection = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [detector, setDetector] = useState<faceDetection.FaceDetector | null>(null);

    useEffect(() => {
        const loadModel = async () => {
            try {
                // Register TensorFlow.js backends
                // await tf.setBackend('webgl'); // Attempt to use WebGL
                await tf.ready(); // Ensure TensorFlow.js is ready

                // console.log('TensorFlow.js backend initialized:', tf.getBackend());

                // Load the MediaPipe Face Detector model
                const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
                const detectorConfig = {
                    runtime: 'tfjs',
                    // maxFaces: 1,
                };
                const faceDetector = await faceDetection.createDetector(model, detectorConfig);
                setDetector(faceDetector);
            } catch (err) {
                console.error('Error loading the face detection model:', err);
            }
        };

        loadModel();

        // Start video streaming from the webcam
        if (navigator.mediaDevices?.getUserMedia) {
            navigator.mediaDevices
                .getUserMedia({ video: true })
                .then((stream) => {
                    if (videoRef.current) {
                        videoRef.current.srcObject = stream;
                        videoRef.current.play();
                    }
                })
                .catch((err) => console.error('Error accessing webcam:', err));
        }
    }, []);

    const detectFaces = async () => {
        if (!detector || !videoRef.current) return;

        const video = videoRef.current;

        try {
            const predictions = await detector.estimateFaces(video, { flipHorizontal: true });
            console.info(predictions);
            if (!predictions || predictions.length === 0) {
                console.warn('No faces detected. Check lighting or camera angle.');
            }

            if (canvasRef.current) {
                const canvas = canvasRef.current;
                const ctx = canvas.getContext('2d');

                if (!ctx) return;

                // Set canvas dimensions to match the video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Clear the canvas for new drawings
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                predictions.forEach((prediction) => {
                    console.info(prediction.box);
                    // const { topLeft, bottomRight, keypoints } = prediction.boundingBox;
                    const box = prediction.box;
                    const keypoints = prediction.keypoints;
                    // const [x, y] = topLeft as [number, number];
                    // const [x2, y2] = bottomRight as [number, number];

                    // Draw bounding box
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 2;
                    // ctx.strokeRect(x, y, x2 - x, y2 - y);
                    ctx.strokeRect(box.xMin, box.yMin, box.width, box.height);

                    // Draw keypoints
                    if (keypoints) {
                        keypoints.forEach((keypoint) => {
                            ctx.fillStyle = 'blue';
                            ctx.fillRect(keypoint.x, keypoint.y, 5, 5);
                        });
                    }
                });
            }
        } catch (err) {
            console.error('Error during face detection:', err);
        }
    };

    useEffect(() => {
        const interval = setInterval(detectFaces, 100); // Detect faces every 100ms
        return () => clearInterval(interval);
    }, [detector]);

    return (
        <div style={{ position: 'relative' }}>
            <video ref={videoRef} style={{ display: 'block', width: 640, height: 480 }} />
            <canvas
                ref={canvasRef}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                }}
            />
        </div>
    );
};

export default FaceDetection;
