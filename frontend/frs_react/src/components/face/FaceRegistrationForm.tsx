"use client"

import React, { useEffect, useRef, useState } from "react";
import { load, MediaPipeFaceDetectorTfjs } from "@/dnn/face_detector/detector";
import { setupBackend } from "@/dnn/tf-backend";
import { registerFace } from "@/api/clients/faceRecognitionClient";
import { drawDetection } from "@/dnn/shared/drawing_utils";
import { Card, CardContent } from "@/components/ui/card";
import { cropFace } from "@/dnn/shared/vision";
import { useRouter, useSearchParams } from "next/navigation";
import { Person } from "@/models/person";


const FaceRegistrationForm = () => {
    const router = useRouter();
    const searchParams = useSearchParams();
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [netDetectionTf, setNetDetectionTf] = useState<MediaPipeFaceDetectorTfjs | null>(null);    

    const person: Person = {
        personId: 123,
        name: `${searchParams.get("fullname")}`,
        email: `${searchParams.get("email")}`,
        age: 30,
        phone: `0111222`,                
        city: 'Dhaka',
        country: 'Bangladesh',
        address: 'Dhaka'
    };

    useEffect(() => {
        // Load TensorFlow model
        const loadModel = async () => {
            await setupBackend();

            const model = await load();
            setNetDetectionTf(model);
        };

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

        loadModel().then(() => {
            startVideo();
        });

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

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Capture video frame as ImageData
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

        const predictions = await netDetectionTf.estimateFaces(video, { width: video.videoWidth, height: video.videoHeight }, { flipHorizontal: false });

        if (predictions.length > 0) {
            // console.log(predictions);            
            for (let i = 0; i < predictions.length; i++) {
                const detection = predictions[i];

                // draw over canvas for visualization
                drawDetection(ctx, detection);

                // crop image based on bounding bbox
                const croppedFace = await cropFace(video, detection.box);
                // send cropped image python backend
                const response = await registerFace(croppedFace, person);
                console.log(response);
            }
        }
    };


    useEffect(() => {
        const interval = setInterval(detectFrame, 100); // Detect faces every 100ms
        return () => clearInterval(interval);
    }, [netDetectionTf]);

    return (
        <Card>
            <CardContent className="flex aspect-square items-center justify-center p-6">
                <video
                    ref={videoRef}
                    style={{
                        display: "block",
                        width: "auto", height: "80vh"
                    }}
                />
                <canvas
                    ref={canvasRef}
                    style={{
                        position: "absolute",
                        display: "block",
                        width: "auto",
                        height: "80vh",
                    }}
                />
            </CardContent>
        </Card>
    );
};

export default FaceRegistrationForm;
