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
import { Person } from "@/models/person";
import { useTensorFlowModel } from "../tensorflow/TensorflowContext";


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
    const [faceLoginWorker, setFaceLoginWorker] = useState<Worker | null>(null);
    const [frsStatus, setFrsStatus] = useState(false);
    const [person, setPerson] = useState<Person | null>(null);


    const { model } = useTensorFlowModel();

    // const faceLoginworker = new Worker('/workers/faceloginworker.ts');


    const stopVideo = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            console.log(`stopping video...`);
            (videoRef.current.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
            videoRef.current.srcObject = null;
            console.log(`video stopped!`);
        }
    }

    const detectFrame = async () => {
        if (frsStatus === true) return;

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

        setIsProcessing(true);

        if (isLoading) {
            setIsLoading(false);            
        }

        try {
            // Set canvas dimensions to match the video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Capture video frame as ImageData
            // ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

            const predictions = await netDetectionTf.estimateFaces(video, { width: video.videoWidth, height: video.videoHeight }, { flipHorizontal: false });

            if (predictions.length > 0) {

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // console.log(predictions);            
                for (let i = 0; i < predictions.length; i++) {
                    const detection = predictions[i];

                    // draw over canvas for visualization
                    drawDetection(ctx, detection);

                    if (frsStatus) {
                        return true;
                    }

                    // crop image based on bounding bbox
                    const croppedFace = await cropFace(video, detection.box);

                    if (!faceLoginWorker) {
                        console.log(`Web worker undefined...`);
                        return false;
                    }

                    faceLoginWorker.postMessage({ blob: croppedFace });

                    faceLoginWorker.onmessage = (e) => {
                        if (e.data.status === true) {
                            setFrsStatus(true);
                            // Parse JSON string into an object
                            const personData: Person = JSON.parse(e.data.result);
                            setPerson(personData);
                        }
                    };
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
            // console.log(`frs status: ${frsStatus}`);
            if (frsStatus === true) {
                // Stop further frame rendering
                cancelAnimationFrame(animationFrameRef.current!);

                // Terminate the worker
                faceLoginWorker?.terminate();

                // Simulate successful face enrollment
                setShowTimer(true);

                // Start the timer countdown
                const timerInterval = setInterval(() => {
                    setTimer((prev) => {
                        if (prev <= 1) {
                            clearInterval(timerInterval);
                            return 0;
                        }
                        return prev - 1;
                    });
                }, 1000);
                return;
            }

            // apply face detection
            const status = await detectFrame();
            if (status !== true) {
                animationFrameRef.current = requestAnimationFrame(renderFrame);
            }
        } catch (ex) {
            console.error(`Error when rendering frame: ${ex}`);
        }
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
        console.log(`trying to initialize worker`);
        if (typeof window !== "undefined") {
            console.log(`entering worker creation...`);
            const worker = new Worker(new URL('@/workers/faceloginworker.ts', import.meta.url));
            // worker.postMessage({ preload: true }); // Custom message to preload necessary scripts
            setFaceLoginWorker(worker);
        }

        return () => {
            faceLoginWorker?.terminate();
        };
    }, []);


    useEffect(() => {
        if (model) {
            setNetDetectionTf(model);
        }
    }, [model]);


    useEffect(() => {
        const initVideo = async () => {
            await startVideo();
        };
        initVideo();
        setIsLoading(false);
        return () => stopVideo();
    }, []);


    useEffect(() => {
        const delayDetection = setTimeout(() => {
            if (netDetectionTf) {
                animationFrameRef.current = requestAnimationFrame(renderFrame);
            }
        }, 1000); // Delay detection for 1 second

        return () => {
            clearTimeout(delayDetection);
            cancelAnimationFrame(animationFrameRef.current!);
        }
    }, [netDetectionTf, frsStatus]);

    // Navigate to the login page when the timer reaches 0
    useEffect(() => {
        if (timer === 0) {
            stopVideo();
            cancelAnimationFrame(animationFrameRef.current!);
            setIsProcessing(false);

            // Terminate the worker
            if (faceLoginWorker) {
                faceLoginWorker.terminate();
            }

            const query = new URLSearchParams(person);
            router.push(`/dashboard?${query}`);
        }
    }, [timer, router, faceLoginWorker, person]);


    // Cleanup on component unmount
    useEffect(() => {
        return () => {
            stopVideo();
            cancelAnimationFrame(animationFrameRef.current!);
            faceLoginWorker?.terminate();
        };
    }, [faceLoginWorker]);

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
                                <h2 className="text-white text-xl font-bold mb-4">Face Match Found!</h2>
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
