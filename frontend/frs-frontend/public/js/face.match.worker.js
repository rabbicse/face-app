//! [Define frames processing]
async function captureFrameAsync() {
    return new Promise(async function (resolve, reject) {
        let frame = undefined;
        let frameBGR = undefined;
        try {
            $("#frsProgress").show();

            await sleep(1);

            //! [Open a camera stream]
            if (cap === undefined) {
                cap = new cv.VideoCapture(camera);
            }

            frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
            frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);

            // Read a frame from camera
            cap.read(frame);
            cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

            var begin = Date.now();

            // If process frame success
            if (await processFrameAsync(frame, frameBGR)) {
                let result = await matchBlob();
                isRunning = !result;
            }

            // Loop this function.
            if (isRunning) {
                var delay = 1000 / FPS - (Date.now() - begin);
                setTimeout(captureFrameAsync, delay);
                resolve(true);
                return;
            }

            resolve(true);
        } catch (exp) {
            console.log("Error when capture frame! Details: ", exp);
            cap = undefined;
            showMessage("Unexpected error when capture from camera! Please reload page!");
            reject(exp);
        } finally {
            $("#frsProgress").hide();
        }
    });
};
//! [Define frames processing]

//! [Process opencv frame asynchronously]
async function processFrameAsync(frame, frameBGR) {
    return new Promise(async function (resolve) {
        try {
            console.time('processFrameAsync');
            cv.imshow(output, frame);

            let faces = undefined;
            if (faceDetectionMethod === "CV") { // detect face via opencv dnn
                faces = await detectFacesByCvAsync(frameBGR);
            } else if (faceDetectionMethod === "TF") { // detect faces by tensorflow
                faces = await detectFacesByTfAsync(frame);
            }

            if (faces === undefined || faces.length <= 0) {
                console.log("No face detected!");
                showMessage("No face detected from client!");
                resolve(false);
                return;
            }

            // If more than 1 face detected then return
            if (faces.length > 1) {
                console.log("Multiple faces detected!");
                showMessage("Multiple faces detected!");
                resolve(false);
                return;
            }

            // clone original frame
            let oFrame = frame.clone();

            // parse results
            let parsedResult = undefined;
            if (faceDetectionMethod === "CV") { // detect face via opencv dnn
                parsedResult = parseCvFaceDetectionResults(faces, oFrame);
            } else if (faceDetectionMethod === "TF") { // detect faces by tensorflow
                parsedResult = parsetfFaceDetectionResults(faces, oFrame);
            }

            if (parsedResult === undefined) {
                showMessage("Couldn't parse face detection results!");
                resolve(false);
                return;
            }
            let croppedFace = parsedResult.face;
            let normRect = parsedResult.rect;

            // check mask detection
            let hasMask = false;
            // let faceMaskResults = await detectMaskAsync(frame);
            // if (faceMaskResults !== undefined && faceMaskResults.length > 0) {
            //     for (result of faceMaskResults.values()) {
            //         console.log(result);
            //         let classID = result[1];
            //         let score = result[2];
            //         console.log('Mask => ', 'class id: ', classID, ' score: ', score);
            //         if (classID === 0 && score > 0.9) {
            //             hasMask = true;
            //             break;
            //         }
            //     }
            // }

            if (hasMask) {
                showMessage("Face mask detected!!!");
                resolve(false);
                return;
            }

            // // todo
            // var rec = faces[0].bbox;
            // // var xp = rec.x;
            // // var yp = rec.y;

            // // // You can try more different parameters
            // // let rc = new cv.Rect(Math.floor(rec.x),
            // //     Math.floor(rec.y),
            // //     Math.ceil(rec.width),
            // //     Math.ceil(rec.height));
            // // let dst = frame.roi(rc);

            // // let normalizedFace = // todo
            // let rectBox = rectFromBox(rec);
            // let normRect = transformNormalizedRect(rectBox, { width: frame.cols, height: frame.rows });
            // // console.log("Normalized rect: ", normRect);

            // let nx1 = Math.max(normRect.xCenter - normRect.width / 2, 0);
            // let ny1 = Math.max(normRect.yCenter - normRect.height / 2, 0);
            // let nx2 = Math.min(nx1 + normRect.width, frame.cols);
            // let ny2 = Math.min(ny1 + normRect.height, frame.rows);
            // cv.rectangle(frame,
            //     { x: Math.floor(nx1), y: Math.floor(ny1) },
            //     { x: Math.floor(nx2), y: Math.floor(ny2) },
            //     [0, 255, 0, 255],
            //     2);
            // let nRect = new cv.Rect(Math.floor(nx1),
            //     Math.floor(ny1),
            //     Math.ceil(nx2 - nx1),
            //     Math.ceil(ny2 - ny1));
            // let dst1 = oFrame.roi(nRect);

            // // console.log(results);
            // // for (bboxInfo of results.values()) {
            // //     bbox = bboxInfo[0];
            // //     classID = bboxInfo[1];
            // //     score = bboxInfo[2];
            // //     console.log("class ID: " + classID);

            // //     if (classID === 0) {
            // //         hasMask = true;
            // //         break;
            // //     }
            // // }

            // // extract face landmark
            // normRect.xCenter = (nx2 - nx1) / 2;
            // normRect.yCenter = (ny2 - ny1) / 2;
            // normRect.width = nx2 - nx1;
            // normRect.height = ny2 - ny1;
            let lmrks = await detectFaceLandmark(croppedFace, normRect);




            // for (const landmark of lmrks) {
            //     cv.circle(frame, {
            //         x: nx1 + Math.floor(landmark.x),
            //         y: ny1 + Math.floor(landmark.y)
            //     }, 1, [0, 255, 0, 255], -1); // nose  
            // }


            // if (lmrks === undefined) {
            //     resolve(false);
            //     return;
            // }

            // let landmarks = [];
            // for (i = 0; i < lmrks.length; i++) {
            //     let lm = lmrks[i];
            //     landmarks.push(lm);
            // }

            // let nose = landmarks[1];
            // let leftEye = landmarks[33];
            // let rightEye = landmarks[263];
            // let leftLip = landmarks[61];
            // let rightLip = landmarks[308];
            // let chick = landmarks[152];


            // // process landmarks
            // let lm = {
            //     nose: { x: nose[0], y: nose[1] },
            //     chin: { x: chick[0], y: chick[1] },
            //     leftEye: { x: leftEye[0], y: leftEye[1] },
            //     rightEye: { x: rightEye[0], y: rightEye[1] },
            //     leftLip: { x: leftLip[0], y: leftLip[1] },
            //     rightLip: { x: rightLip[0], y: rightLip[1] }
            // }

            // if (drawOverlay) {
            //     drawAllFaceLandmarks(frame, landmarks, xp, yp);
            // }

            if (!estimatePose(lmrks, normRect)) {
                showMessage("Face orientation not perfect!");
                resolve(false);
                return;
            }


            // faces.forEach(function (rect) {
            //     // draw over processed frame
            //     let x = Math.floor(rect.x);
            //     let y = Math.floor(rect.y);
            //     let w = Math.ceil(rect.width);
            //     let h = Math.ceil(rect.height);

            //     cv.rectangle(frame,
            //         { x: x, y: y },
            //         { x: x + w, y: y + h },
            //         [0, 255, 0, 255],
            //         2);
            // });

            // let faceRect = faces[0];
            // let xTh = rect.width / 3;
            // let yTh = rect.height / 2;
            // let x = Math.max(rect.x - xTh, 0);
            // let y = Math.max(rect.y - yTh, 0);
            // let x1 = Math.min(rect.x + rect.width + xTh, frame.cols);
            // let y1 = Math.min(rect.y + rect.height + yTh, frame.rows);
            // let rct = new cv.Rect(x, y, x1 - x, y1 - y);
            // // console.log(rct);
            // let face = oFrame.roi(rct);
            // // face = cv.cvtColor(face, cv.COLOR_RGBA2BGR);

            // let face = cropFaceFromDetectedFace(oFrame, faceRect);
            // cv.imshow(faceCanvas, face);


            try {
                cv.imshow(output, frame);
            } catch (exp) {
                console.log("Error when show frame to canvas!");
            }

            resolve(false);
        } catch (exp) {
            console.log(exp);
            resolve(false);
        } finally {
            // frame.delete();
            // frameBGR.delete();
            console.timeEnd('processFrameAsync');
        }
    });
}