const faceDetectionCvProtoUrl = '/models/detections/face_detection_cv.prototxt';
const faceDetectionCvWeightUrl = '/models/detections/face_detection_cv.caffemodel';
const faceDetectionCvProtoPath = 'face_detector.prototxt';
const faceDetectionCvWeightsPath = 'face_detector.caffemodel';

const faceLandmarkTfProtoUrl = '/models/landmarks/model.json';
const faceLandmarkTfWeightUrl = '/models/landmarks/group1-shard1of1.bin';

const faceDetectionTfProtoUrl = '/models/detections/model.json';
const faceDetectionTfWeightUrl = '/models/detections/group1-shard1of1.bin';

const faceMaskDetectionTfProtoUrl = '/models/mask/model.json';
const faceMaskDetectionTfWeightUrl = '/models/mask/group1-shard1of1.bin';

const opencvUrl = '/js/opencv.js';
const opencvKey = 'opencv.js';

const tensorflowUrl = '/js/tf.min.js';
const tensorflowKey = 'tf.min.js';



let netDet = undefined;
let netMaskTf = undefined;
let netLandmarkTf = undefined;
let netDetectionTf = undefined;


//! [Run face detection model]
function detectFacesAsync(img) {
    return new Promise(function (resolve, reject) {
        let faces = [];
        let blob = undefined;
        let out = undefined;
        try {
            blob = cv.blobFromImage(img, 1, { width: 192, height: 144 }, [104, 117, 123, 0], false, false);
            netDet.setInput(blob);
            out = netDet.forward();

            for (var i = 0, n = out.data32F.length; i < n; i += 7) {
                var confidence = out.data32F[i + 2];
                var left = out.data32F[i + 3] * img.cols;
                var top = out.data32F[i + 4] * img.rows;
                var right = out.data32F[i + 5] * img.cols;
                var bottom = out.data32F[i + 6] * img.rows;
                left = Math.min(Math.max(0, left), img.cols - 1);
                right = Math.min(Math.max(0, right), img.cols - 1);
                bottom = Math.min(Math.max(0, bottom), img.rows - 1);
                top = Math.min(Math.max(0, top), img.rows - 1);

                if (confidence > 0.5 && left < right && top < bottom) {
                    faces.push({ x: left, y: top, width: right - left, height: bottom - top })
                }
            }

            if (faces.length < 1) {
                resolve(faces);
                return;
            }

            resolve(faces);

        } catch (ex) {
            console.log("Error when apply face detection: ", ex);
            reject(ex);
        } finally {
            blob.delete();
            out.delete();
        }
    });
};
//! [Run face detection model]

//! [Run face detection model]
function detectFaceTfAsync(img) {
    return new Promise(async function (resolve) {
        try {
            let anchorsData = generateAnchors(128, 128, ANCHORS_CONFIG);
            let anchors = tf.tensor(anchorsData);
            let inputSize = tf.tensor([128, 128]);

            // get scale factor
            let scaleFactor = tf.div([img.cols, img.rows], tf.tensor([128, 128]));

            // convert to RGB    
            let rgb = new cv.Mat(img.cols, img.rows, cv.CV_8UC3);
            cv.cvtColor(img, rgb, cv.COLOR_BGRA2RGB);

            let [detectedOutputs, boxes, scores] = tf.tidy(() => {
                // create tensor from rgb image
                let tensor = tf.tensor(rgb.data, [rgb.rows, rgb.cols, rgb.channels()]);

                // resize 
                let inputTensor = tf.image.resizeBilinear(tensor, [128, 128]);
                inputTensor = tf.mul(tf.sub(tf.div(inputTensor, 255), 0.5), 2);
                inputTensor = inputTensor.expandDims(0).toFloat();


                let batchPrediction = netDetectionTf.predict(inputTensor);
                let prediction = tf.squeeze(batchPrediction);


                let decodedBounds = decodeBounds(prediction, anchors, inputSize);

                const logits = tf.slice(prediction, [0, 0], [-1, 1]);
                const scores = tf.squeeze(tf.sigmoid(logits));

                return [prediction, decodedBounds, scores];
            });

            // console.log("Logits: ", logits.dataSync());
            // console.log("Score: ", scores.dataSync());
            // console.log("Decoded Bounds: ", decodedBounds.dataSync());


            const boxIndicesTensor = await tf.image.nonMaxSuppressionAsync(boxes, scores, 10, 0.3, 0.75);

            // console.log("Box indices tensor: ", boxIndicesTensor);

            const boxIndices = await boxIndicesTensor.array();
            let boundingBoxes = boxIndices.map(
                (boxIndex) => tf.slice(boxes, [boxIndex, 0], [1, -1]));

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

                    let landmarks = tf.reshape(tf.squeeze(tf.slice(detectedOutputs,
                        [boxIndex, NUM_LANDMARKS - 1], [1, -1])), [NUM_LANDMARKS, -1]);
                    let normalizedLandmarks = tf.mul(tf.add(landmarks, anchor), scaleFactor);

                    // console.log("Annotate: ", { bbox, lm, pro });
                    return { bbox, normalizedLandmarks, probability, anchor };
                });


                // let lm = landmarks.dataSync();                
                const landmarkData = await Promise.all([annotatedBox.normalizedLandmarks].map(async d => d.array()));
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

            // console.log("Annotated box: ", annotatedBoxes);

            resolve(annotatedBoxes);

        } catch (ex) {
            console.log("Error when apply face detection with tensorflow!");
            console.log(ex);
            resolve(undefined);
        }
    });
};
//! [Run face detection model]


//! [Run face detection model]
// https://creativetech.blog/home/face-landmarks-for-arcore-augmented-faces
async function detectFaceLandmark(img, normRect) {
    return new Promise(async function (resolve, reject) {
        try {
            // convert to RGB    
            let rgb = new cv.Mat(img.rows, img.cols, cv.CV_8UC3);
            cv.cvtColor(img, rgb, cv.COLOR_BGRA2RGB);

            let outputTensors = tf.tidy(() => {
                // tensor from original image
                let tensor = tf.tensor(rgb.data, [rgb.rows, rgb.cols, rgb.channels()]);

                // resized tensor 
                let inputTensor = tf.image.resizeBilinear(tensor, [192, 192]);

                inputTensor = tf.mul(tf.sub(tf.div(inputTensor, 255), 0.5), 2);
                inputTensor = inputTensor.expandDims(0).toFloat();

                const outputs = ['output_faceflag'].concat(['output_mesh']);
                return netLandmarkTf.execute(inputTensor, outputs);
            });

            let faceFlagTensor = outputTensors[0];
            let landmarkTensors = outputTensors[1];


            const facePresenceScore = (await faceFlagTensor.data())[0];
            console.log("Face present: ", facePresenceScore);

            const landmarks = await tensorsToLandmarks(landmarkTensors);
            // console.log("Landmarks: ", landmarks);

            const finalLandmark = calculateLandmarkProjection(landmarks, normRect);
            // console.log("Final Landmarks: ", finalLandmark);

            for (const landmark of finalLandmark) {
                cv.circle(img, {
                    x: Math.floor(landmark.x),
                    y: Math.floor(landmark.y)
                }, 1, [0, 255, 0, 255], -1); // nose  
            }

            cv.imshow(faceMeshCanvas, img);


            resolve(finalLandmark);

        } catch (ex) {
            console.log("Error when apply face landmark detection!");
            console.log(ex);
            reject(ex);
        }
    });
};
//! [Run face detection model]

//! [face mask detection]
async function detectMaskAsync(img) {
    let anchors = anchorGenerator(featureMapSizes, anchorSizes, anchorRatios);
    // convert to RGB    
    let rgb = new cv.Mat(img.rows, img.cols, cv.CV_8UC3);
    cv.cvtColor(img, rgb, cv.COLOR_BGRA2RGB);

    const [rawBBoxes, rawConfidences] = tf.tidy(() => {
        // tensor from original image
        let tensor = tf.tensor(rgb.data, [rgb.rows, rgb.cols, rgb.channels()]);

        // resized tensor 
        let inputTensor = tf.image.resizeBilinear(tensor, [260, 260]);

        inputTensor = tf.mul(tf.sub(tf.div(inputTensor, 255), 0.5), 2);
        inputTensor = inputTensor.expandDims(0).toFloat();

        const [rawBBoxes, rawConfidences] = netMaskTf.predict(inputTensor);
        return [rawBBoxes, rawConfidences];
    });

    const bboxes = decodeBBox(anchors, tf.squeeze(rawBBoxes));
    return await nonMaxSuppression(bboxes, tf.squeeze(rawConfidences), 0.5, 0.5, rgb.cols, rgb.rows);    
};
//! [face mask detection]


//! [Initialize Cache]
async function initializeCache() {
    return new Promise(async function (resolve, reject) {
        try {

            // check if tensorflow key exists
            let tfData = await getModelByName(dbName, dbVersion, storeName, 'model_name', tensorflowKey);
            // if tensorflow not cached then cache tensorflow library to indexed db
            if (tfData === undefined) {
                // download tensorflow js and cache to indexed db
                let response = await downloadTextFileAsync(tensorflowKey, tensorflowUrl);
                if (response === undefined) {
                    showMessage("tf download failed!");
                    reject("tf download failed!");
                    return;
                }

                // insert tensorflow js to indexed db
                insertModel(dbName, dbVersion, storeName, { "model_name": tensorflowKey, "model": response });

                // inject tensorflow js inside DOM
                document.getElementById("tensorflow").innerHTML = response;

            } else {
                // inject tensorflow js inside DOM
                document.getElementById("tensorflow").innerHTML = tfData['model'];
            }



            // check if opencv key exists
            let opencvData = await getModelByName(dbName, dbVersion, storeName, 'model_name', opencvKey);
            // Download proto file for caffe model
            if (opencvData === undefined) {
                let response = await downloadTextFileAsync(opencvKey, opencvUrl);

                if (response === undefined) {
                    showMessage("opencv download failed!");
                    reject("opencv download failed!");
                    return;
                }

                // insert opencv js to indexed db
                insertModel(dbName, dbVersion, storeName, { "model_name": opencvKey, "model": response });

                // console.log("opencv downloaded...");
                document.getElementById("opencv").innerHTML = response;
            } else {
                document.getElementById("opencv").innerHTML = opencvData['model'];
            }

            resolve(true);

        } catch (err) {
            console.log("Error when load opencv");
            console.log(err);
            reject(err);
        }
    });
}
//! [Initialize Cache]


//! [Initialize DNN]
async function initializeDnn() {
    return new Promise(async function (resolve, reject) {
        try {
            let faceDetectionProtoData = await getModelByName(dbName, dbVersion, storeName, 'model_name', faceDetectionCvProtoPath);
            // console.log("face detection proto data: ", faceDetectionProtoData);

            // Download proto file for caffe model
            if (faceDetectionProtoData === undefined) {
                let protoDownloadStatus = await downloadFileAsync(faceDetectionCvProtoPath, faceDetectionCvProtoUrl, true);
                // console.log("Proto downloaded status: ", protoDownloadStatus);
                if (protoDownloadStatus !== undefined) {
                    saveDnnToFile(faceDetectionCvProtoPath, protoDownloadStatus);
                }
            } else {
                saveDnnToFile(faceDetectionProtoData['model_name'], new Int8Array(faceDetectionProtoData['model']));
            }

            let faceDetectionWeightData = await getModelByName(dbName, dbVersion, storeName, 'model_name', faceDetectionCvWeightsPath);
            // console.log("face detection caffe model data: ", faceDetectionWeightData);

            if (faceDetectionWeightData === undefined) {
                let weightDownloadStatus = await downloadFileAsync(faceDetectionCvWeightsPath, faceDetectionCvWeightUrl, true);
                // console.log("Caffemodel downloaded status: ", weightDownloadStatus);
                if (weightDownloadStatus !== undefined) {
                    saveDnnToFile(faceDetectionCvWeightsPath, weightDownloadStatus);
                }
            } else {
                saveDnnToFile(faceDetectionWeightData['model_name'], new Int8Array(faceDetectionWeightData['model']));
            }


            // console.log("face detection cv model loaded...");




            // load tensorflow js models
            let tfMaskInfo = await getTfModelByName('tensorflowjs', dbVersion, 'model_info_store', 'tf-landmark-model');
            console.log("tf landmark info: ", tfMaskInfo);

            let tfMaskData = await getTfModelByName('tensorflowjs', dbVersion, 'models_store', 'tf-landmark-model');
            console.log("tf landmark data: ", tfMaskData);


            if (tfMaskInfo === undefined || tfMaskData === undefined) {
                console.log("downloading tf models");
                netLandmarkTf = await tf.loadGraphModel(faceLandmarkTfProtoUrl);
                console.log("face landmark net loaded...");
                await netLandmarkTf.save('indexeddb://tf-landmark-model');
            } else {
                netLandmarkTf = await tf.loadGraphModel('indexeddb://tf-landmark-model');
            }


            // load detector
            await initFaceDetectorTfModel();

            // load face mask detector
            netMaskTf = await initFaceMaskTfModel();

            resolve(true);
        } catch (err) {
            console.log("Error when load models");
            console.log(err);
            reject(err);
        }
    });
}
//! [Initialize DNN]

async function initFaceDetectorTfModel() {
    // load tensorflow js models
    let tfMaskInfo = await getTfModelByName('tensorflowjs', dbVersion, 'model_info_store', 'tf-detector-model');
    console.log("tf landmark info: ", tfMaskInfo);

    let tfMaskData = await getTfModelByName('tensorflowjs', dbVersion, 'models_store', 'tf-detector-model');
    console.log("tf landmark data: ", tfMaskData);


    if (tfMaskInfo === undefined || tfMaskData === undefined) {
        console.log("downloading tf models");
        netDetectionTf = await tf.loadGraphModel(faceDetectionTfProtoUrl);
        console.log("face detector net loaded...");
        await netDetectionTf.save('indexeddb://tf-detector-model');
    } else {
        netDetectionTf = await tf.loadGraphModel('indexeddb://tf-detector-model');
    }
}

async function initFaceMaskTfModel() {
    let net = undefined;
    // load tensorflow js models
    let tfMaskInfo = await getTfModelByName('tensorflowjs', dbVersion, 'model_info_store', 'tf-mask-model');
    console.log("tf mask info: ", tfMaskInfo);

    let tfMaskData = await getTfModelByName('tensorflowjs', dbVersion, 'models_store', 'tf-mask-model');
    console.log("tf mask data: ", tfMaskData);


    if (tfMaskInfo === undefined || tfMaskData === undefined) {
        console.log("downloading tf maskmodels");
        net = await tf.loadLayersModel(faceMaskDetectionTfProtoUrl);
        console.log("face mask detector net loaded...");
        await net.save('indexeddb://tf-mask-model');
    } else {
        net = await tf.loadLayersModel('indexeddb://tf-mask-model');
    }

    return net;
}

async function initOpencvNet() {
    return new Promise(function (resolve) {
        try {
            netDet = cv.readNetFromCaffe(faceDetectionCvProtoPath, faceDetectionCvWeightsPath);
            resolve(true);
        } catch (err) {
            console.log("Error when opencv net. ", err);
            resolve(false);
        }
    });
}

async function downloadTfModelAsync(path, uri) {
    try {
        return $.ajax({
            method: "GET",
            url: uri,
            timeout: 0,
            success: function (response) {
                console.log("download success...");
                let data = new Uint8Array(response);

                // insert model to index db
                insertModel(dbName, dbVersion, storeName, { "model_name": path, "model": data });
            },
            error: function (err) {
                console.log(err);
            }
        });
    } catch (ex) {
        console.log(ex);
    }
}

function saveDnnToFile(path, data) {
    cv.FS_createDataFile('/', path, data, true, false, false);
}

function estimatePose(landmarkData, size) {

    try {
        let nose = landmarkData[1];
        let leftEye = landmarkData[33];
        let rightEye = landmarkData[263];
        let leftLip = landmarkData[61];
        let rightLip = landmarkData[308];
        let chick = landmarkData[152];


        // process landmarks
        let landmarks = {
            nose: { x: nose.x, y: nose.y },
            chin: { x: chick.x, y: chick.y },
            leftEye: { x: leftEye.x, y: leftEye.y },
            rightEye: { x: rightEye.x, y: rightEye.y },
            leftLip: { x: leftLip.x, y: leftLip.y },
            rightLip: { x: rightLip.x, y: rightLip.y }
        }

        // console.log(landmarks);



        const numRows = 6;
        const modelPoints = cv.matFromArray(numRows, 3, cv.CV_64FC1, [
            0.0,
            0.0,
            0.0, // Nose tip

            0.0,
            -330.0,
            -65.0, // Chin

            -165.0,
            170.0,
            -135.0, // Left eye left corner

            165.0,
            170.0,
            -135.0, // Right eye right corner

            -150.0,
            -150.0,
            -125.0, // Left Mouth corner

            150.0,
            -150.0,
            -125.0 // Right mouth corner
        ]);


        // Camera internals
        // const size = { width: im.cols, height: im.rows };
        // const focalLength = size.width;
        const center = [size.width / 2, size.height / 2];
        const focalLength = center[0] / Math.tan(60 / 2 * Math.PI / 180)
        const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
            focalLength,
            0,
            center[0],
            0,
            focalLength,
            center[1],
            0,
            0,
            1
        ]);

        // Create Matrixes
        let imagePoints = cv.Mat.zeros(numRows, 2, cv.CV_64FC1);
        let distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
        let rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
        let tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
        let jaco = new cv.Mat();
        let R = new cv.Mat();
        let jaco2 = new cv.Mat();


        // // 2D image points. If you change the image, you need to change vector
        // [
        //     landmarks.nose.x * size.width,
        //     landmarks.nose.y * size.height, // Nose tip

        //     landmarks.chin.x * size.width,
        //     landmarks.chin.y * size.height, // Nose tip

        //     landmarks.leftEye.x * size.width,
        //     landmarks.leftEye.y * size.height, // Nose tip

        //     landmarks.rightEye.x * size.width,
        //     landmarks.rightEye.y * size.height, // Nose tip

        //     landmarks.leftLip.x * size.width,
        //     landmarks.leftLip.y * size.height, // Nose tip

        //     landmarks.rightLip.x * size.width,
        //     landmarks.rightLip.y * size.height, // Nose tip
        // ].map((v, i) => {
        //     imagePoints.data64F[i] = v;
        // });


        // 2D image points. If you change the image, you need to change vector
        [
            landmarks.nose.x,
            landmarks.nose.y, // Nose tip

            landmarks.chin.x,
            landmarks.chin.y, // Nose tip

            landmarks.leftEye.x,
            landmarks.leftEye.y, // Nose tip

            landmarks.rightEye.x,
            landmarks.rightEye.y, // Nose tip

            landmarks.leftLip.x,
            landmarks.leftLip.y, // Nose tip

            landmarks.rightLip.x,
            landmarks.rightLip.y, // Nose tip
        ].map((v, i) => {
            imagePoints.data64F[i] = v;
        });

        const success = cv.solvePnP(
            modelPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs,
            rvec,
            tvec,
            false,
            cv.SOLVEPNP_ITERATIVE
        );

        if (success) {
            let axis = cv.matFromArray(3, 3, cv.CV_64FC1, [500, 0.0, 0.0, 0.0, 500, 0.0, 0.0, 0.0, 500.0]);
            let modelPointss = new cv.Mat();
            cv.projectPoints(
                axis,
                rvec,
                tvec,
                cameraMatrix,
                distCoeffs,
                modelPointss,
                jaco
            );

            let modelPoints2 = new cv.Mat();
            cv.projectPoints(
                modelPoints,
                rvec,
                tvec,
                cameraMatrix,
                distCoeffs,
                modelPoints2,
                jaco2
            );
            cv.Rodrigues(rvec, R);


            // test with formula
            let sq = Math.sqrt((R.data64F[0] * R.data64F[0]) + (R.data64F[3] * R.data64F[3]))
            let x = Math.atan2(R.data64F[7], R.data64F[8]);
            let y = Math.atan2(-R.data64F[6], sq);
            let z = Math.atan2(R.data64F[3], R.data64F[0]);

            let rollF = (z / Math.PI) * 180;
            let pitchF = (x / Math.PI) * 180;
            let yawF = (y / Math.PI) * 180;

            pitchF = Math.asin(Math.sin(pitchF * Math.PI / 180)) * 180 / Math.PI;
            rollF = -Math.asin(Math.sin(rollF * Math.PI / 180)) * 180 / Math.PI;
            yawF = Math.asin(Math.sin(yawF * Math.PI / 180)) * 180 / Math.PI;

            console.log('Roll: ', rollF, 'Pitch: ', pitchF, 'Yaw: ', yawF);
            return Math.abs(rollF) <= 20 && Math.abs(pitchF) <= 20 && Math.abs(yawF) <= 20;
        }
    } catch (err) {
        console.log(err);
    }

    return false;
}
