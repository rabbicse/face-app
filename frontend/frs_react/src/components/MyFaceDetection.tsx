"use client"

import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";

// num of landmarks during face detection with blazeface
const NUM_LANDMARKS = 6;
// anchors config during face detection on 
const ANCHORS_CONFIG = {
    'strides': [8, 16],
    'anchors': [2, 6]
};

// blazeface detection config
const config = {
    rotation: 0,
    rotationDegree: 0,
    shiftX: 0,
    shiftY: 0,
    squareLong: true,
    scaleX: 1.5,
    scaleY: 1.5
}

const featureMapSizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]];
const anchorSizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]];
const anchorRatios = [[1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42]];

function generateAnchors(width, height, outputSpec) {
    const anchors = [];
    for (let i = 0; i < outputSpec.strides.length; i++) {
        const stride = outputSpec.strides[i];
        const gridRows = Math.floor((height + stride - 1) / stride);
        const gridCols = Math.floor((width + stride - 1) / stride);
        const anchorsNum = outputSpec.anchors[i];

        for (let gridY = 0; gridY < gridRows; gridY++) {
            const anchorY = stride * (gridY + 0.5);

            for (let gridX = 0; gridX < gridCols; gridX++) {
                const anchorX = stride * (gridX + 0.5);
                for (let n = 0; n < anchorsNum; n++) {
                    anchors.push([anchorX, anchorY]);
                }
            }
        }
    }

    return anchors;
}

function decodeBounds(boxOutputs, anchors, inputSize) {
    var boxStarts = tf.slice(boxOutputs, [0, 1], [-1, 2]);
    var centers = tf.add(boxStarts, anchors);
    var boxSizes = tf.slice(boxOutputs, [0, 3], [-1, 2]);
    var boxSizesNormalized = tf.div(boxSizes, inputSize);
    var centersNormalized = tf.div(centers, inputSize);
    var halfBoxSize = tf.div(boxSizesNormalized, 2);
    var starts = tf.sub(centersNormalized, halfBoxSize);
    var ends = tf.add(centersNormalized, halfBoxSize);
    var startNormalized = tf.mul(starts, inputSize);
    var endNormalized = tf.mul(ends, inputSize);
    var concatAxis = 1;
    return tf.concat2d([startNormalized, endNormalized], concatAxis);
}



// async function detectFaceTfAsync(imgData, netDetectionTf) {
//     // num of landmarks during face detection with blazeface
//     const NUM_LANDMARKS = 6;
//     // anchors config during face detection on 
//     const ANCHORS_CONFIG = {
//         'strides': [8, 16],
//         'anchors': [2, 6]
//     };
//     // const ANCHORS_CONFIG = {}; // Provide your anchors configuration
//     // const NUM_LANDMARKS = 68; // Example: Adjust based on your model
//     const inputSize = [128, 128]; // Model input size

//     try {
//         const anchorsData = generateAnchors(inputSize[0], inputSize[1], ANCHORS_CONFIG);
//         const anchors = tf.tensor(anchorsData);

//         // Convert image to Tensor

//         // Capture the current video frame
//         const imgTensor = tf.tidy(() => {
//             const videoFrame = tf.browser.fromPixels(imgData); // Create a tensor from video frame
//             const resizedFrame = tf.image.resizeBilinear(videoFrame, inputSize); // Resize for model
//             const normalizedImg = tf.mul(tf.sub(tf.div(resizedFrame, 255), 0.5), 2).expandDims(0);
//             return normalizedImg;
//         });

//         // const imgTensor = tf.browser.fromPixels(imgData);
//         // const resizedImg = tf.image.resizeBilinear(imgTensor, inputSize);
//         // const normalizedImg = tf.mul(tf.sub(tf.div(resizedImg, 255), 0.5), 2).expandDims(0);

//         const batchPrediction = netDetectionTf.predict(imgTensor);
//         const prediction = tf.squeeze(batchPrediction);

//         const decodedBounds = decodeBounds(prediction, anchors, tf.tensor(inputSize));
//         const logits = tf.slice(prediction, [0, 0], [-1, 1]);
//         const scores = tf.squeeze(tf.sigmoid(logits));

//         const boxIndicesTensor = await tf.image.nonMaxSuppressionAsync(
//             decodedBounds,
//             scores,
//             10,
//             0.3,
//             0.75
//         );

//         const boxIndices = await boxIndicesTensor.array();
//         const boundingBoxes = boxIndices.map((boxIndex) =>
//             tf.slice(decodedBounds, [boxIndex, 0], [1, -1])
//         );

//         const results = await Promise.all(
//             boundingBoxes.map(async (boundingBox, i) => {
//                 const boxArray = await boundingBox.array();
//                 boundingBox.dispose();

//                 const landmarks = tf.slice(
//                     prediction,
//                     [boxIndices[i], NUM_LANDMARKS - 1],
//                     [1, -1]
//                 );
//                 const landmarksArray = await landmarks.array();
//                 landmarks.dispose();

//                 const probability = (await tf.slice(scores, [boxIndices[i]], [1]).array())[0];

//                 return {
//                     bbox: {
//                         x: boxArray[0][0],
//                         y: boxArray[0][1],
//                         width: boxArray[0][2] - boxArray[0][0],
//                         height: boxArray[0][3] - boxArray[0][1],
//                     },
//                     landmarks: landmarksArray,
//                     probability,
//                 };
//             })
//         );

//         console.log(results);

//         return results.filter((result) => result.probability > 0.95);
//     } catch (err) {
//         console.error("Error during face detection:", err);
//         return [];
//     }
// }

function createBox(startEndTensor) {
    return {
        startEndTensor,
        startPoint: tf.slice(startEndTensor, [0, 0], [-1, 2]),
        endPoint: tf.slice(startEndTensor, [0, 2], [-1, 2])
    }
};

async function detectFaceTfAsync(imgData: HTMLVideoElement, netDetectionTf) {
    // num of landmarks during face detection with blazeface
    const NUM_LANDMARKS = 6;
    const inputSize = [128, 128]; // Model input size

    try {
        // generate anchors
        const anchorsData = generateAnchors(inputSize[0], inputSize[1], ANCHORS_CONFIG);
        const anchors = tf.tensor(anchorsData);

        // get scale factor
        let scaleFactor = tf.div([imgData.videoWidth, imgData.videoHeight], tf.tensor([128, 128]));

        // Capture the current video frame
        const imgTensor = tf.tidy(() => {
            // capture video frame to tf tensor
            const videoFrame = tf.browser.fromPixels(imgData); // Create a tensor from video frame

            // resize frame
            const resizedFrame = tf.image.resizeBilinear(videoFrame, inputSize); // Resize for model

            // normalize tensor
            const normalizedImg = tf.mul(tf.sub(tf.div(resizedFrame, 255), 0.5), 2).expandDims(0).toFloat();
            return normalizedImg;
        });


        // Predict tensor
        // Pass tensor to fit with model
        const batchPrediction = netDetectionTf.predict(imgTensor);
        const prediction = tf.squeeze(batchPrediction);

        // decode predictions
        // decode bounding boxes
        const decodedBounds = decodeBounds(prediction, anchors, tf.tensor(inputSize));

        // decode prediction scores
        const logits = tf.slice(prediction, [0, 0], [-1, 1]);
        const scores = tf.squeeze(tf.sigmoid(logits));

        // Apply nms
        const boxIndicesTensor = await tf.image.nonMaxSuppressionAsync(decodedBounds, scores, 10, 0.3, 0.75);

        // decode bounding boxes
        const boxIndices = await boxIndicesTensor.array();
        let boundingBoxes = boxIndices.map((boxIndex) =>
            tf.slice(decodedBounds, [boxIndex, 0], [1, -1])
        );


        // console.log("Boxes: ", boundingBoxes.array());
        boundingBoxes = await Promise.all(
            boundingBoxes.map(async (boundingBox) => {
                const vals = await boundingBox.array();
                boundingBox.dispose();
                return vals;
            }));



        let annotatedBoxes = [];
        let returnTensors = true;
        for (let i = 0; i < boundingBoxes.length; i++) {
            const boxIndex = boxIndices[i];
            let probability = tf.slice(scores, [boxIndex], [1]);
            if (probability < 0.95) {
                continue;
            }

            const boundingBox = boundingBoxes[i];
            const annotatedBox = tf.tidy(() => {
                const box = boundingBox instanceof tf.tensor ?
                    createBox(boundingBox) :
                    createBox(tf.tensor(boundingBox));


                let start = tf.mul(box.startPoint, scaleFactor).dataSync();
                let end = tf.mul(box.endPoint, scaleFactor).dataSync();
                let bbox = {
                    x: start[0],
                    y: start[1],
                    width: end[0] - start[0],
                    height: end[1] - start[1]
                };

                let anchor;
                if (returnTensors) {
                    anchor = tf.slice(anchors, [boxIndex, 0], [1, 2]);
                } else {
                    anchor = anchorsData[boxIndex];
                }

                let landmarks = tf.reshape(tf.squeeze(tf.slice(prediction,
                    [boxIndex, NUM_LANDMARKS - 1], [1, -1])), [NUM_LANDMARKS, -1]);
                let normalizedLandmarks = tf.mul(tf.add(landmarks, anchor), scaleFactor);

                // console.log("Annotate: ", { bbox, lm, pro });
                return { bbox, normalizedLandmarks, probability, anchor };
            });


            // let lm = landmarks.dataSync();                
            const landmarkData = await Promise.all([annotatedBox.normalizedLandmarks].map(async d => await d.array()));
            const scaledLandmark = landmarkData.map(landmark => ([
                (landmark[0] + annotatedBox.anchor[0]) * scaleFactor,
                (landmark[1] + annotatedBox.anchor[1]) * scaleFactor
            ]));

            let pro = await probability.data();
            pro = pro[0];
            let anc = await annotatedBox.anchor.data();
            anc = anc[0];

            // push annotations to array
            annotatedBoxes.push({
                bbox: annotatedBox.bbox,
                landmarks: scaledLandmark,
                probablility: pro,
                anchor: anc
            });
        }

        return annotatedBoxes;


        // const results = await Promise.all(
        //     boundingBoxes.map(async (boundingBox, i) => {
        //         const boxArray = await boundingBox.array();
        //         boundingBox.dispose();

        //         const landmarks = tf.slice(
        //             prediction,
        //             [boxIndices[i], NUM_LANDMARKS - 1],
        //             [1, -1]
        //         );
        //         const landmarksArray = await landmarks.array();
        //         landmarks.dispose();

        //         const probability = (await tf.slice(scores, [boxIndices[i]], [1]).array())[0];

        //         return {
        //             bbox: {
        //                 x: boxArray[0][0],
        //                 y: boxArray[0][1],
        //                 width: boxArray[0][2] - boxArray[0][0],
        //                 height: boxArray[0][3] - boxArray[0][1],
        //             },
        //             landmarks: landmarksArray,
        //             probability,
        //         };
        //     })
        // );

        // console.log(results);

        // return results.filter((result) => result.probability > 0.95);
    } catch (err) {
        console.error("Error during face detection:", err);
        return [];
    }
}


const MyFaceDetection = () => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [netDetectionTf, setNetDetectionTf] = useState<tf.GraphModel | null>(null);

    useEffect(() => {
        // Load TensorFlow model
        const loadModel = async () => {
            const model = await tf.loadGraphModel("/models/detection/model.json");
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

        // Capture video frame as ImageData
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
        // const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height);

        // if (!imageData) return;
        // const videoFrame = tf.browser.fromPixels(video); // Create a tensor from video frame

        // Run face detection
        const results = await detectFaceTfAsync(video, netDetectionTf);
        console.log(results);

        // Draw results
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
        ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);
        results.forEach((result) => {
            const { bbox, landmarks } = result;

            // Draw bounding box
            ctx?.beginPath();
            ctx?.rect(bbox.x, bbox.y, bbox.width, bbox.height);
            ctx!.lineWidth = 2;
            ctx!.strokeStyle = "red";
            ctx?.stroke();

            // Draw landmarks
            landmarks.forEach(([x, y]) => {
                ctx?.beginPath();
                ctx?.arc(x, y, 3, 0, 2 * Math.PI);
                ctx!.fillStyle = "blue";
                ctx?.fill();
            });
        });

        // requestAnimationFrame(detectFrame);
    };

    // useEffect(() => {
    //     if (netDetectionTf) detectFrame();
    // }, [netDetectionTf]);

    useEffect(() => {
        const interval = setInterval(detectFrame, 100); // Detect faces every 100ms
        return () => clearInterval(interval);
    }, [netDetectionTf]);

    return (
        <div style={{ position: "relative" }}>
            <video
                ref={videoRef}
                style={{ display: "block", width: "640", height: 480 }}
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

export default MyFaceDetection;
