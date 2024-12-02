"use client"

import FaceDetection from "@/components/FaceDetection";
import Image from "next/image";

export default function Home() {
  return (
    <div>
      <h1>Face Detection with TensorFlow.js</h1>
      <FaceDetection />
    </div>
  );
}
