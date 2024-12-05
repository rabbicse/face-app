"use client"

import FaceDetection from "@/components/FaceDetection";
import FaceDetectionTfjs from "@/components/FaceDetector";
import MyFaceDetection from "@/components/MyFaceDetection";

export default function Home() {
  return (
    <div>
      <h1>Face Detection with TensorFlow.js</h1>
      <MyFaceDetection />
    </div>
  );
}
