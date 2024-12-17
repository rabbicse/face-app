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
import { FaceRegResponse } from "@/models/responses";


const FaceRegistrationForm = () => {
    const router = useRouter();
    const searchParams = useSearchParams();
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [netDetectionTf, setNetDetectionTf] = useState<MediaPipeFaceDetectorTfjs | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const animationFrameRef = useRef<number | null>(null);

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

    const stopVideo = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            console.log(`stopping video...`);
            (videoRef.current.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
            videoRef.current.srcObject = null;
            console.log(`video stopped!`);
        }
    }

    const detectFrame = async () => {
        if (!videoRef.current || !canvasRef.current || !netDetectionTf) {
            console.log(`videoref: ${videoRef.current} canvasref: ${canvasRef.current} net: ${netDetectionTf}`);
            return false;
        }

        const video = videoRef.current;
        if (video.videoWidth == 0 || video.videoHeight == 0) {
            console.log(`video wxh: ${video.width} x ${video.height}`);
            return false;
        }

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (!ctx) {
            console.log(`CTX: ${ctx}`);
            return false;
        }

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
                const response: FaceRegResponse = await registerFace(croppedFace, person);
                console.log(response.status);
                if (response.status == 0) {
                    return true;
                }
            }
        }
        return false;
    };

    const renderFrame = async () => {
        setIsProcessing(true);
        try {
            console.log(`rendering frame...`);
            let result = await detectFrame();
            console.log(result);
            if (result === true) {
                stopVideo();
                cancelAnimationFrame(animationFrameRef.current!);
                return;
            }
            animationFrameRef.current = requestAnimationFrame(renderFrame);
        } catch (ex) {
            console.error(`Error when rendering frame: ${ex}`);
        } finally {
            setIsProcessing(false);
        }
    };


    // Load TensorFlow model
    const loadModel = async () => {
        await setupBackend();

        const model = await load();
        setNetDetectionTf(model);
        console.log(`model: ${model}`);
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
                console.log(`Metadata loaded...`);
                videoRef.current?.play();
            };
        }
    };

    useEffect(() => {
        loadModel()
            .then(() => {
                startVideo();
            });

        return stopVideo();
    }, []);


    useEffect(() => {
        if (netDetectionTf) {
            animationFrameRef.current = requestAnimationFrame(renderFrame);
        }
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
