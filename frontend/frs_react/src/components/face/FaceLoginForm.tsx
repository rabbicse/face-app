"use client"

import React, { useEffect, useRef, useState } from "react";
import { load, MediaPipeFaceDetectorTfjs } from "@/dnn/face_detector/detector";
import { setupBackend } from "@/dnn/tf-backend";
import { loginByFace } from "@/api/clients/faceRecognitionClient";
import { drawDetection } from "@/dnn/shared/drawing_utils";
import { Card, CardContent } from "@/components/ui/card";
import { cropFace } from "@/dnn/shared/vision";
import { useRouter, useSearchParams } from "next/navigation";
import { FaceRegResponse } from "@/models/responses";
import { Loader2 } from "lucide-react";


const FaceLoginForm = () => {
    const router = useRouter();
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [netDetectionTf, setNetDetectionTf] = useState<MediaPipeFaceDetectorTfjs | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const [timer, setTimer] = useState(5); // Timer countdown
    const [showTimer, setShowTimer] = useState(false); // Toggle to show timer UI
    const animationFrameRef = useRef<number | null>(null);


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
            // console.log(`videoref: ${videoRef.current} canvasref: ${canvasRef.current} net: ${netDetectionTf}`);
            return false;
        }

        const video = videoRef.current;
        if (video.videoWidth == 0 || video.videoHeight == 0) {
            // console.log(`video wxh: ${video.width} x ${video.height}`);
            return false;
        }

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (!ctx) {
            // console.log(`CTX: ${ctx}`);
            return false;
        }

        try {
            // Set canvas dimensions to match the video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Capture video frame as ImageData
            ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

            setIsProcessing(true);

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
                    const response: FaceRegResponse = await loginByFace(croppedFace);
                    if (response != null && response.status == 0) {
                        return true;
                    }
                }
            }
        } catch (ex) {
            console.log(`Error when process frame! Details: ${ex}`)
        } finally {
            setIsProcessing(false);
        }
        return false;
    };

    const renderFrame = async () => {
        try {
            console.log(`rendering frame...`);
            let result = await detectFrame();
            console.log(result);
            if (result === true) {

                // Simulate successful face enrollment
                setShowTimer(true);

                // Start the timer countdown
                const timerInterval = setInterval(() => {
                    setTimer((prev) => {
                        if (prev <= 1) {
                            clearInterval(timerInterval);
                            cancelAnimationFrame(animationFrameRef.current!);
                            return 0;
                        }
                        return prev - 1;
                    });
                }, 1000);
                return;
            }
            animationFrameRef.current = requestAnimationFrame(renderFrame);
        } catch (ex) {
            console.error(`Error when rendering frame: ${ex}`);
        }
    };


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
                console.log(`Metadata loaded...`);
                videoRef.current?.play();
            };
        }
    };

    useEffect(() => {
        startVideo();
        loadModel().then(() => setIsLoading(false));
    }, []);


    useEffect(() => {
        if (netDetectionTf) {
            animationFrameRef.current = requestAnimationFrame(renderFrame);
        }
    }, [netDetectionTf]);

    // Navigate to the login page when the timer reaches 0
    useEffect(() => {
        if (timer === 0) {
            stopVideo();
            router.push("/dashboard");
        }
    }, [timer, router]);

    return (
        <Card>
            <CardContent className="relative flex aspect-square items-center justify-center p-6">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center">
                        <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
                        <p className="mt-2 text-gray-600">Initializing...</p>
                    </div>
                ) : (
                    <>
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
                                display: isProcessing ? "block" : "none",
                                width: "auto",
                                height: "80vh",
                            }}
                        />

                        {/* Timer Animation */}
                        {showTimer && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-50">
                                <div className="text-white text-xl font-bold mb-4">
                                    Redirecting in {timer} seconds...
                                </div>
                                {/* Circular Progress Animation */}
                                <div className="w-16 h-16 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
                            </div>
                        )}
                    </>
                )}
            </CardContent>
        </Card>
    );
};

export default FaceLoginForm;
