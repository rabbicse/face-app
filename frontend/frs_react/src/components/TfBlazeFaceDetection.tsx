"use client"

import React, { useEffect, useRef, useState } from "react";
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from "@tensorflow/tfjs";
import { load } from "@/lib/blazeface/blaze";
import { BlazeFaceModel } from "@/lib/blazeface/face";

const TfBlazeFaceDetection = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [netDetectionTf, setNetDetectionTf] = useState<BlazeFaceModel | null>(null);

    useEffect(() => {
        // Load TensorFlow model
        const loadModel = async () => {
            await tf.setBackend("cpu");

            const model = await load();
            setNetDetectionTf(model);
        };
        loadModel();

        // Initialize video stream
        const startVideo = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                // videoRef.current.play();
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
        // const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height);

        // if (!imageData) return;
        // const videoFrame = tf.browser.fromPixels(video); // Create a tensor from video frame

        // Run face detection
        // const results = await detectFaceTfAsync(video, netDetectionTf);
        // console.log(results);

        // // Draw results
        // ctx?.clearRect(0, 0, canvas.width, canvas.height);
        // ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
        // results.forEach((result) => {
        //     const { bbox, landmarks } = result;

        //     // Draw bounding box
        //     ctx?.beginPath();
        //     ctx?.rect(bbox.x, bbox.y, bbox.width, bbox.height);
        //     ctx!.lineWidth = 2;
        //     ctx!.strokeStyle = "red";
        //     ctx?.stroke();

        //     // Draw landmarks
        //     landmarks.forEach(([x, y]) => {
        //         ctx?.beginPath();
        //         ctx?.arc(x, y, 3, 0, 2 * Math.PI);
        //         ctx!.fillStyle = "blue";
        //         ctx?.fill();
        //     });
        // });

        const returnTensors = false;
        const flipHorizontal = true;
        const annotateBoxes = true;
        const predictions = await netDetectionTf.estimateFaces(
            video, returnTensors, flipHorizontal, annotateBoxes);
        console.log(`predictions length: ${predictions.length}`);

        if (predictions.length > 0) {
            console.log(predictions);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (let i = 0; i < predictions.length; i++) {
                console.log(`prediction[${i}] => ${predictions}`)
                // if (returnTensors) {

                //     console.log(predictions[i].topLeft.arraySync());

                //     predictions[i].topLeft = predictions[i].topLeft.arraySync();
                //     predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
                //     if (annotateBoxes) {
                //         predictions[i].landmarks = predictions[i].landmarks.arraySync();
                //     }
                // }

                const start = predictions[i].topLeft;
                const end = predictions[i].bottomRight;
                console.log(start);
                console.log(end);

                console.log(`start: ${start} end: ${end}`);

                const size = [Math.abs(end[0]) - Math.abs(start[0]), Math.abs(end[1]) - Math.abs(start[1])];
                console.log(`size: ${size}`);
                ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
                ctx.fillRect(Math.abs(start[0]), start[1], size[0], size[1]);

                if (annotateBoxes) {
                    const landmarks = predictions[i].landmarks;

                    ctx.fillStyle = 'blue';
                    for (let j = 0; j < landmarks.length; j++) {
                        const x = Math.abs(landmarks[j][0]);
                        const y = Math.abs(landmarks[j][1]);
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
                style={{ display: "block", width: 640, height: 480 }}
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

export default TfBlazeFaceDetection;
