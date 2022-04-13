let isRunning = true;
let output = undefined;
let camera_output = undefined;
let faceCanvas = undefined;
let faceMeshCanvas = undefined;
let toast = undefined;

//! [On document ready]
function onDocumentReady() {
    $(document).ready(async function () {
        try {
            $("#frsProgress").hide();
            // Update elements
            output = document.getElementById('output');
            camera_output = document.getElementById('camera_output');
            faceCanvas = document.getElementById("face");
            faceMeshCanvas = document.getElementById("face_mesh");

            // Create a camera object.
            camera = document.getElementById("video");//document.createElement("video");


            // Toast
            var toastElList = [].slice.call(document.querySelectorAll('.toast'))
            var toastList = toastElList.map(function (toastEl) {
                return new bootstrap.Toast(toastEl)
            });
            toast = toastList[0];

            let isCacheInitialized = await initializeCache();
            if (!isCacheInitialized) {
                showMessage("Error when initialize cache!");
                return;
            }

            await processWebcamAsync(camera, loadDnnAsync);
        } catch (exp) {
            console.log("Error when load page: Details: ", exp);
        } finally {
            // Show hide
            $("#progress").hide();
        }
    });
}
//! [document ready]

async function loadDnnAsync() {
    return new Promise(async function (resolve, reject) {
        try {
            $("#progress").show();
            // load opencv if exists inside indexeddb        
            let isDnnLoaded = await initializeDnn();
            if (!isDnnLoaded) {
                reject("Dnn didn't loaded!");
                return;
            }

            let isDnnInitialized = await initOpencvNet();
            if (!isDnnInitialized) {
                reject("Dnn didn't initialized!");
                return;
            }

            setTimeout(captureFrameAsync, 1);

            resolve(true); // todo
        } finally {
            $("#progress").hide();
        }
    });
}


function showMessage(msg) {
    $('#status').html(msg);
    toast.show();
}

//! [Define frames processing]
async function captureFrameAsync() {
    return new Promise(async function (resolve, reject) {
        try {
            $("#frsProgress").show();

            await sleep(1);

            //! [Open a camera stream]
            var cap = new cv.VideoCapture(camera);
            var frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
            var frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);
            //! [Open a camera stream]

            cap.read(frame);  // Read a frame from camera
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
            reject(exp);
        } finally {
            $("#frsProgress").hide();
        }
    });
};
//! [Define frames processing]

// const faceDetectionMethod = "CV";
const faceDetectionMethod = "TF";
const drawOverlay = true;

//! [Process opencv frame asynchronously]
async function processFrameAsync(frame, frameBGR) {
    return new Promise(async function (resolve) {
        try {
            console.log("frs progress on...");

            // clone original frame
            let oFrame = frame.clone();

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

            // todo
            var rec = faces[0].bbox;
            var xp = rec.x;
            var yp = rec.y;

            // You can try more different parameters
            let rc = new cv.Rect(Math.floor(rec.x),
                Math.floor(rec.y),
                Math.ceil(rec.width),
                Math.ceil(rec.height));
            let dst = frame.roi(rc);

            // let normalizedFace = // todo
            let rectBox = rectFromBox(rec);
            let normRect = transformNormalizedRect(rectBox, { width: frame.cols, height: frame.rows });
            console.log("Normalized rect: ", normRect);

            let nx1 = Math.max(normRect.xCenter - normRect.width / 2, 0);
            let ny1 = Math.max(normRect.yCenter - normRect.height / 2, 0);
            let nx2 = Math.min(nx1 + normRect.width, frame.cols);
            let ny2 = Math.min(ny1 + normRect.height, frame.rows);
            cv.rectangle(frame,
                { x: Math.floor(nx1), y: Math.floor(ny1) },
                { x: Math.floor(nx2), y: Math.floor(ny2) },
                [0, 255, 0, 255],
                2);
            let nRect = new cv.Rect(Math.floor(nx1),
                Math.floor(ny1),
                Math.ceil(nx2 -nx1),
                Math.ceil(ny2 - ny1));
            let dst1 = oFrame.roi(nRect);

            console.log("ok...");

            // todo

            normRect.xCenter = (nx2 - nx1) / 2;
            normRect.yCenter = (ny2 - ny1) / 2;
            normRect.width = nx2 - nx1;
            normRect.height = ny2 - ny1;
            let lmrks = await detectFaceLandmark(dst1, normRect);
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

            if (!estimatePose(lmrks, { width: dst1.cols, height: dst1.rows })) {
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

            resolve(false);
        } catch (exp) {
            console.log(exp);
            resolve(false);
        } finally {
            cv.imshow(output, frame);

            frame.delete();
            frameBGR.delete();
        }
    });
}

async function detectFacesByCvAsync(frame) {
    return await detectFacesAsync(frame);
}

async function detectFacesByTfAsync(frame) {
    return await detectFaceTfAsync(frame);
}

function cropFaceFromDetectedFace(frame, rect) {
    let xTh = rect.width / 3;
    let yTh = rect.height / 2;
    let x = Math.max(rect.x - xTh, 0);
    let y = Math.max(rect.y - yTh, 0);
    let x1 = Math.min(rect.x + rect.width + xTh, frame.cols);
    let y1 = Math.min(rect.y + rect.height + yTh, frame.rows);
    let rct = new cv.Rect(x, y, x1 - x, y1 - y);

    return frame.roi(rct);
}

function drawFaceLandmarks(frame, landmarks, xp, yp, rec) {
    let nose = landmarks[1];
    let leftEye = landmarks[33];
    let rightEye = landmarks[263];
    let leftLip = landmarks[61];
    let rightLip = landmarks[308];
    let chick = landmarks[152];

    console.log({ x: xp + leftEye[0], y: yp + leftEye[1] });



    // cv.circle(frame, { x: xp + nose[0] * rec.width, y: yp + nose[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose
    // cv.circle(frame, { x: xp + leftEye[0] * rec.width, y: yp + leftEye[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose
    // cv.circle(frame, { x: xp + rightEye[0] * rec.width, y: yp + rightEye[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose
    // cv.circle(frame, { x: xp + leftLip[0] * rec.width, y: yp + leftLip[1] * rec.height }, 3, [255, 0, 0, 255], -1); // nose
    // cv.circle(frame, { x: xp + rightLip[0] * rec.width, y: yp + rightLip[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose
    // cv.circle(frame, { x: xp + chick[0] * rec.width, y: yp + chick[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose


    cv.circle(frame, { x: xp + nose[0], y: yp + nose[1] }, 3, [0, 255, 0, 255], -1); // nose
    cv.circle(frame, { x: xp + leftEye[0], y: yp + leftEye[1] }, 3, [0, 255, 0, 255], -1); // nose
    cv.circle(frame, { x: xp + rightEye[0], y: yp + rightEye[1] }, 3, [0, 255, 0, 255], -1); // nose
    cv.circle(frame, { x: xp + leftLip[0], y: yp + leftLip[1] }, 3, [255, 0, 0, 255], -1); // nose
    cv.circle(frame, { x: xp + rightLip[0], y: yp + rightLip[1] }, 3, [0, 255, 0, 255], -1); // nose
    cv.circle(frame, { x: xp + chick[0], y: yp + chick[1] }, 3, [0, 255, 0, 255], -1); // nose
}

function drawAllFaceLandmarks(frame, landmarks, xp, yp) {
    landmarks.forEach(function (lm) {
        cv.circle(frame, { x: xp + lm[0], y: yp + lm[1] }, 1, [0, 255, 0, 255], -1); // nose
    });
}


//! [Match blob from canvas]
async function matchBlob() {
    try {
        return true;
        // get name from input
        let name = $("#name").val();
        // get blob from canvas
        let blob = await getCanvasBlob(faceCanvas);

        // if no blob data found then return
        if (blob === null || blob === undefined) {
            showMessage("No detected frame found!");
            return;
        }

        // create form data object to call service
        const formData = new FormData();
        formData.append("image", blob, "photo.jpg");
        formData.append("data", JSON.stringify({ "name": name }));

        let response = await recognizeFace(formData);
        if (response.status != 0) {
            // todo message
            return false;
        }

        // parse result
        return isFaceMatched(response.result);
    } catch (exp) {
        console.log("Error when match blob! Details: ", exp);
    }
}
//! [Match blob from canvas]

function isFaceMatched(response) {
    try {
        var parsedJson = $.parseJSON(response);
        var matchScore = parsedJson["score"];

        return matchScore >= 0.65;
    } catch (exp) {
        console.error("Error when match face! Details: ", exp);
    }

    return false;
}
