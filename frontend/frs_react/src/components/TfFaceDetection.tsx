"use client"

import React, { useEffect, useRef, useState } from "react";
import { load, MediaPipeFaceDetectorTfjs } from "@/dnn/face_detector/detector";
import { setupBackend } from "@/dnn/tf-backend";
import { BoundingBox } from "@/dnn/shared/interfaces/shapes";
import { sendCroppedFace } from "@/api/clients/faceRecognitionClient";
import { drawDetection } from "@/dnn/shared/drawing_utils";
import { Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious } from "@/components/ui/carousel";
import { Card, CardContent } from "./ui/card";

const cropFace = (image: HTMLVideoElement, box: BoundingBox): Promise<Blob> => {
    return new Promise((resolve) => {
        // Create a temporary canvas to crop the face
        const faceCanvas = document.createElement("canvas");
        const faceCtx = faceCanvas.getContext("2d");

        // const xMin = Math.max(Math.abs(box.xMin) - (box.width / 4), 0);
        // const yMin = Math.max(Math.abs(box.yMin) - (box.height / 4), 0);
        // const xMax = Math.min(Math.abs(box.xMax) + (box.width / 4) + (box.width / 4), image.width);
        // const yMax = Math.min(Math.abs(box.yMax) + (box.height / 8) + (box.height / 4), image.height);
        // const width = Math.abs(xMax - xMin);
        // const height = Math.abs(yMax - yMin);

        const xMin = Math.max(box.xMin - (box.width / 4), 0);
        const yMin = Math.max(Math.abs(box.yMin) - (box.height / 2), 0);
        const width = box.width + (2 * (box.width / 4));// + (box.width / 4) + (box.width / 4); //Math.abs(xMax - xMin);
        const height = box.height + ((box.height / 2) + (box.height / 4)); //Math.abs(yMax - yMin);

        faceCanvas.width = width;
        faceCanvas.height = height;

        // Draw the cropped face onto the temporary canvas
        faceCtx?.drawImage(
            image, // The main canvas containing the video
            xMin,
            yMin,
            width,
            height,
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
    const [carouselItems, setCarouselItems] = useState([]);
    const [src, setSrc] = useState(''); // initial src will be empty

    const addCarouselItem = (blob: Blob) => {
        const imageUrl = URL.createObjectURL(blob); // Convert blob to URL
        setCarouselItems((prevItems) => {
            const updatedItems = [...prevItems, { id: Date.now(), image: imageUrl }];
            // Keep only the last 10 items
            if (updatedItems.length > 10) {
                const removedItem = updatedItems.shift(); // Remove the first (oldest) item
                if (removedItem) {
                    URL.revokeObjectURL(removedItem.image); // Clean up old blob URL
                }
            }
            return updatedItems;
        });
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
                addCarouselItem(croppedFace);
                // send cropped image python backend
                // const response = await sendCroppedFace(croppedFace);
                // console.log(response);
            }
        }
    };


    useEffect(() => {
        const interval = setInterval(detectFrame, 100); // Detect faces every 100ms
        return () => clearInterval(interval);
    }, [netDetectionTf]);

    return (
        <>
            <div style={{ display: "flex", width: "100%", height: "90vh" }}>

                <div style={{ flex: 1, position: "relative", width: "100%", height: "80%" }}>
                    <div className="p-1">
                        <Card>
                            <CardContent className="flex aspect-square items-center justify-center p-6">
                                <video
                                    ref={videoRef}
                                    style={{
                                        display: "none",
                                        width: "auto", height: "100%"
                                    }}
                                />
                                <canvas
                                    ref={canvasRef}
                                    style={{
                                        position: "absolute",
                                        display: "block",
                                        width: "auto",
                                        height: "100%",
                                    }}
                                />

                            </CardContent>
                        </Card>
                    </div>
                </div>

                <div style={{
                    flex: 1,
                    display: "flex",
                    width: "40%",
                    justifyContent: "center",
                    alignItems: "center",
                    backgroundColor: "#f4f4f4"
                }}>
                    <Carousel className="w-full max-w-xs">
                        <CarouselContent>
                            {carouselItems.map((item, index) => (
                                <CarouselItem key={index} className="flex aspect-square items-center justify-center p-6">
                                    <div className="p-1">
                                        <Card>
                                            <CardContent className="flex aspect-square items-center justify-center p-6">
                                                <img src={item.image} alt={`Carousel Item ${index + 1}`} />
                                            </CardContent>
                                        </Card>
                                    </div>
                                </CarouselItem>
                            ))}
                        </CarouselContent>
                        <CarouselPrevious />
                        <CarouselNext />
                    </Carousel>
                </div>
            </div>
        </>
    );
};

export default TfFaceDetection;
