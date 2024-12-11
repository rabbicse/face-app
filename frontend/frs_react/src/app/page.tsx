"use client"

// import TfBlazeFaceDetection from "@/components/TfBlazeFaceDetection";
import TfFaceDetection from "@/components/TfFaceDetection";

// import FaceDetection from "@/components/FaceDetection";
// import FaceDetectionTfjs from "@/components/FaceDetector";
// import MyFaceDetection from "@/components/MyFaceDetection";


export default function Home() {
  return (
    <div>
      {/* <h1>Face Detection with TensorFlow.js</h1> */}
      {/* <MyFaceDetection /> */}
      {/* <TfBlazeFaceDetection /> */}
      <TfFaceDetection />
    </div>
  );
}
