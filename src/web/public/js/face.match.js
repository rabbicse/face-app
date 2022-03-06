let persons = {};
let isRunning = false;
let isRecognizing = false;

const webCamActive = "Webcam";
const webCamInactive = "Webcam Off";

let output = undefined; // document.getElementById('output');
let camera_output = undefined; // document.getElementById('output');
let faceCanvas = undefined; // document.getElementById("face");
let matchimageElement = undefined;
let matchInputElement = undefined;

let toast = undefined;

//! [On document ready]
function onDocumentReady() {
    $(document).ready(async function () {
        //! [initialize variables based on dom]
        matchInputElement = document.getElementById('matchImage');
        matchimageElement = document.getElementById('matchImageSrc');
        //! [initialize variables based on dom]

        // Show hide
        $("#progress").hide();

        // Toast
        var toastElList = [].slice.call(document.querySelectorAll('.toast'))
        var toastList = toastElList.map(function (toastEl) {
            return new bootstrap.Toast(toastEl)
        });
        toast = toastList[0];


        // Update elements
        output = document.getElementById('output');
        camera_output = document.getElementById('camera_output');
        faceCanvas = document.getElementById("face");
        console.log("opencv is ready...");
    
        // Create a camera object.
        camera = document.createElement("video");





        //! [Add image event listeners]
        addImageEventListeners();
        //! [Add image event listeners]


        // add button events
        //! [add camera start/stop button event] 
        $("#startStopButton").click(function (e) {
            e.preventDefault();

            console.log("Running status: " + isRunning);
            if (isRunning) {
                isRunning = false;
                $('#startStopButton').html(webCamActive);
                stopStreamedVideo(camera);
            } else {
                onMatchFromWebcam();
            }
        });
        //! [add camera start/stop button event]

        //! [match image button event]
        $("#matchImageButton").click(function (e) {
            e.preventDefault();
            matchImage();
        });

        // load opencv if exists inside indexeddb
        await initializeOpenCV();

        showMessage("Dnn models loaded...");
    });
}
//! [document ready]

//! [opencv ready]
async function onOpenCvReady() {
    output = document.getElementById('output');
    camera_output = document.getElementById('camera_output');
    faceCanvas = document.getElementById("face");
    console.log("opencv is ready...");

    // Create a camera object.
    camera = document.createElement("video");
    // camera.setAttribute("width", output.width);
    // camera.setAttribute("height", output.height);

    // load dnn models
    // loadModels(processAsync);
    await initializeDnn();

    showMessage("Dnn models loaded...");
}
//! [opencv ready]

function showMessage(msg) {
    $('#status').html(msg);
    toast.show();
}



//! [Recognize face]
async function recognizeFace(formData) {
    try {
        return $.ajax({
            method: "POST",
            url: "http://localhost:5000/match/v1",
            data: formData,
            processData: false,
            contentType: false,
            timeout: 5000,
            mimeType: "multipart/form-data",
            success: function (result) {
                console.log(result);
                var parsedJson = $.parseJSON(result);
                var matchScore = parsedJson["score"];

                if (matchScore < 0) {
                    showMessage("Please look at the camera...");
                    $("#score").html("Undefined");
                    $("#matchStatus").html("Not Matched");
                    return;
                }

                var score = (parseFloat(matchScore) * 100).toFixed(2) + '%';
                $("#score").html(score);

                if (matchScore >= 0.65) {
                    $("#matchStatus").html("Matched");
                } else {
                    $("#matchStatus").html("Not Matched");
                }
            },
            error: function (err) {
                console.log("Error...");
                console.log(err);
            }
        });
    } catch (e) {
        console.log(e);
    }
}
//! [Recognize face]

//! [Define frames processing]
async function captureFrame() {
    //! [Open a camera stream]
    var cap = new cv.VideoCapture(camera);
    var frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
    var frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);
    //! [Open a camera stream]

    cap.read(frame);  // Read a frame from camera
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    var begin = Date.now();

    // If process frame success
    if (await processFrame(frame, frameBGR)) {
        await matchBlob();
    }

    // Loop this function.
    if (isRunning) {
        var delay = 1000 / FPS - (Date.now() - begin);
        setTimeout(captureFrame, delay);
    }
};
//! [Define frames processing]

//!
async function processFrame(frame, frameBGR) {
    try {
        var faces = detectFaces(frameBGR);
        if (faces.length <= 0) {
            console.log("No face detected!");
            return false;
        }





        // todo
        var rec = faces[0];
        var xp = rec.x;
        var yp = rec.y;
        var x2 = rec.x + rec.width;
        var y2 = rec.y + rec.height;

        let dst = new cv.Mat();
        // You can try more different parameters
        let rc = new cv.Rect(rec.x, rec.y, rec.width, rec.height);
        dst = frame.roi(rc);

        // todo
        let lmrks = detectFaceLandmark(dst)
        // console.log(lmrks);
        let landmarks = [];
        for (i = 0; i < lmrks.length; i++) {
            let lm = lmrks[i];
            landmarks.push(lm);
            // cv.circle(frame, { x: xp + lm[0] * rec.width, y: yp + lm[1] * rec.height }, 1, [0, 255, 0, 255], -1); // nose
        }

        let nose = landmarks[1];
        // let leftEye = landmarks[33]; //landmarks[468]; // landmarks[130]; // stable
        // let rightEye = landmarks[263]; //landmarks[473] // landmarks[359];// ;
        let leftEye = landmarks[33]; //landmarks[468]; // landmarks[130]; // stable
        let rightEye = landmarks[263]; //landmarks[473] // landmarks[359];// ;
        let leftLip = landmarks[61];
        let rightLip = landmarks[308];
        let chick = landmarks[152];

        let lm = {
            nose: { x: nose[0], y: nose[1] },
            chin: { x: chick[0], y: chick[1] },
            leftEye: { x: leftEye[0], y: leftEye[1] },
            rightEye: { x: rightEye[0], y: rightEye[1] },
            leftLip: { x: leftLip[0], y: leftLip[1] },
            rightLip: { x: rightLip[0], y: rightLip[1] }
        }

        estimatePose(lm, dst);

        cv.circle(frame, { x: xp + nose[0] * rec.width, y: yp + nose[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose
        cv.circle(frame, { x: xp + leftEye[0] * rec.width, y: yp + leftEye[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose
        cv.circle(frame, { x: xp + rightEye[0] * rec.width, y: yp + rightEye[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose
        cv.circle(frame, { x: xp + leftLip[0] * rec.width, y: yp + leftLip[1] * rec.height }, 3, [255, 0, 0, 255], -1); // nose
        cv.circle(frame, { x: xp + rightLip[0] * rec.width, y: yp + rightLip[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose
        cv.circle(frame, { x: xp + chick[0] * rec.width, y: yp + chick[1] * rec.height }, 3, [0, 255, 0, 255], -1); // nose


        // let dst = new cv.Mat();
        // // You can try more different parameters
        // let rc = new cv.Rect(rec.x, rec.y, rec.width, rec.height);
        // dst = frame.roi(rc);


        // var landmarks = detectFaceMask(dst);
        // for (i = 0, n = landmarks.data32F.length / 2; i < n; i++) {
        //     console.log(landmarks.data32F[i]);
        //     let x = landmarks.data32F[2 * i] * rec.width + xp;
        //     let y = landmarks.data32F[(2 * i) + 1] * rec.height + yp;

        //     console.log({ "x": x, "y": y });

        //     cv.circle(frame, { x: x, y: y }, 3, [0, 255, 0, 255], -1); // nose
        //     // break;
        // }

        // clone original frame
        let oFrame = frame.clone();

        faces.forEach(function (rect) {
            // draw over processed frame
            cv.rectangle(frame,
                { x: rect.x, y: rect.y },
                { x: rect.x + rect.width, y: rect.y + rect.height },
                [0, 255, 0, 255],
                2);
        });

        cv.imshow(output, frame);

        // If more than 1 face detected then return
        if (faces.length !== 1) {
            return false;
        }

        let rect = faces[0];
        let xTh = rect.width / 5;
        let yTh = rect.height / 5;
        let x = Math.max(rect.x - xTh, 0);
        let y = Math.max(rect.y - yTh, 0);
        let x1 = Math.min(rect.x + rect.width + xTh, frame.cols);
        let y1 = Math.min(rect.y + rect.height + yTh, frame.rows);
        let rct = new cv.Rect(x, y, x1 - x, y1 - y);
        // console.log(rct);
        let face = oFrame.roi(rct);
        // face = cv.cvtColor(face, cv.COLOR_RGBA2BGR);
        cv.imshow(faceCanvas, face);

        return true;
    } catch (exp) {
        console.log(exp);
    } finally {
        frame.delete();
        frameBGR.delete();
    }

    return false;
}


function estimatePose(landmarks, im) {

    try {
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
        const size = { width: im.cols, height: im.rows };
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
        // console.log("Camera Matrix:", cameraMatrix.data64F);

        // Create Matrixes
        let imagePoints = cv.Mat.zeros(numRows, 2, cv.CV_64FC1);
        let distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
        let rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
        let tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
        // let pointZ = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 0.0, 500.0]);
        // let pointY = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 500.0, 0.0]);
        // let pointX = cv.matFromArray(1, 3, cv.CV_64FC1, [500.0, 0.0, 0.0]);
        // let noseEndPoint2DZ = new cv.Mat();
        // let nose_end_point2DY = new cv.Mat();
        // let nose_end_point2DX = new cv.Mat();
        let jaco = new cv.Mat();
        let R = new cv.Mat();
        let jaco2 = new cv.Mat();


        // 2D image points. If you change the image, you need to change vector
        [
            landmarks.nose.x * size.width,
            landmarks.nose.y * size.height, // Nose tip

            landmarks.chin.x * size.width,
            landmarks.chin.y * size.height, // Nose tip

            landmarks.leftEye.x * size.width,
            landmarks.leftEye.y * size.height, // Nose tip

            landmarks.rightEye.x * size.width,
            landmarks.rightEye.y * size.height, // Nose tip

            landmarks.leftLip.x * size.width,
            landmarks.leftLip.y * size.height, // Nose tip

            landmarks.rightLip.x * size.width,
            landmarks.rightLip.y * size.height, // Nose tip
        ].map((v, i) => {
            imagePoints.data64F[i] = v;
        });
        console.log(imagePoints);



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

        // console.log(success);

        if (success) {
            console.log("Rotation Vector:", rvec.data64F);
            // console.log(
            //     "Rotation Vector (in degree):",
            //     rvec.data64F.map(d => (d / Math.PI) * 180));


            // Project a 3D points [0.0, 0.0, 500.0],  [0.0, 500.0, 0.0],
            //   [500.0, 0.0, 0.0] as z, y, x axis in red, green, blue color


            // console.log(cv.projectPoints);

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

            console.log("R: ", R);

            // console.log("R", R.data64F);
            // https://learnopencv.com/rotation-matrix-to-euler-angles/
            // https://coderedirect.com/questions/170082/how-to-calculate-the-angle-from-rotation-matrix
            // https://coderedirect.com/questions/170082/how-to-calculate-the-angle-from-rotation-matrix
            // https://www.joyk.com/dig/detail/1530041405701973

            // let sq = Math.sqrt((R.data64F[7] * R.data64F[7]) + (R.data64F[8] * R.data64F[8]))
            // let x = Math.atan2(R.data64F[7], R.data64F[8]);
            // let y = Math.atan2(-R.data64F[6], sq);
            // let z = Math.atan2(R.data64F[4], R.data64F[0]);
            // console.log("Rotation (x, y, z): ", x, y, z);

            // console.log("Rotation (x, y, z) degree: ", (x / Math.PI) * 180, (y / Math.PI) * 180, (z / Math.PI) * 180);

            // console.log("TVEC", tvec.rows, tvec.cols);

            // console.log("Rotation Vector:", rvec.data64F);
            // let rvDegree = rvec.data64F.map(d => (d / Math.PI) * 180);
            // console.log(
            //     "Rotation Vector (in degree):",
            //     rvDegree
            // );
            // console.log("Translation Vector:", tvec.data64F);

            // // Initialise a MatVector
            // let matVec = new cv.MatVector();
            // // Push a Mat back into MatVector
            // matVec.push_back(R);

            // // let matTvec = new cv.MatVector();
            // // matTvec.push_back(tvec);

            // console.log(tvec.data64F);

            // console.log(matVec.data64F);
            // console.log(tvec.data64F);

            // cv.hconcat(matVec, tvec);
            // console.log(tvec.data64F);




            // test with formula
            let sq = Math.sqrt((R.data64F[0] * R.data64F[0]) + (R.data64F[3] * R.data64F[3]))
            let x = Math.atan2(R.data64F[7], R.data64F[8]);
            let y = Math.atan2(-R.data64F[6], sq);
            let z = Math.atan2(R.data64F[3], R.data64F[0]);
            console.log("Rotation (x, y, z): ", x, y, z);
            console.log("Rotation (x, y, z) degree: ", (x / Math.PI) * 180, (y / Math.PI) * 180, (z / Math.PI) * 180);
            // end test with formula




            let proj = cv.matFromArray(3, 4, cv.CV_64FC1, [
                R.data64F[0],
                R.data64F[1],
                R.data64F[2],
                tvec.data64F[0],
                R.data64F[3],
                R.data64F[4],
                R.data64F[5],
                tvec.data64F[1],
                R.data64F[6],
                R.data64F[7],
                R.data64F[8],
                tvec.data64F[2]
            ]);
            console.log(proj.data64F);




            // https://blog-mahoroi-com.translate.goog/posts/2020/05/browser-head-pose-estimation/?_x_tr_sl=ja&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=sc
            // https://justadudewhohacks.github.io/opencv4nodejs/docs/Mat#decomposeProjectionMatrix
            let cmat = new cv.Mat();
            let rotmat = new cv.Mat();
            let travec = new cv.Mat();
            let rotmatX = new cv.Mat();
            let rotmatY = new cv.Mat();
            let rotmatZ = new cv.Mat();
            let eulerAngles = new cv.Mat();
            cv.decomposeProjectionMatrix(proj,
                cmat,
                rotmat,
                travec,
                rotmatX,
                rotmatY,
                rotmatZ,
                eulerAngles);

            let pitch = eulerAngles.data64F[0];
            let yaw = eulerAngles.data64F[1];
            let roll = eulerAngles.data64F[2];

            console.log('Roll: ', roll, 'Pitch: ', pitch, 'Yaw: ', yaw);
            console.log('Roll: ', roll * Math.PI / 180, 'Pitch: ', pitch * Math.PI / 180, 'Yaw: ', yaw * Math.PI / 180);


            pitch = Math.asin(Math.sin(pitch * Math.PI / 180)) * 180 / Math.PI;
            roll = -Math.asin(Math.sin(roll * Math.PI / 180)) * 180 / Math.PI;
            yaw = Math.asin(Math.sin(yaw * Math.PI / 180)) * 180 / Math.PI;

            console.log('Roll: ', roll, 'Pitch: ', pitch, 'Yaw: ', yaw);
            // console.log('Roll: ', roll * Math.PI / 180, 'Pitch: ', pitch * Math.PI / 180, 'Yaw: ', yaw * Math.PI / 180);

            // let sq = Math.sqrt((tvec.data64F[7] * tvec.data64F[7]) + (tvec.data64F[8] * tvec.data64F[8]))
            // let x = Math.atan2(tvec.data64F[7], tvec.data64F[8]);
            // let y = Math.atan2(-tvec.data64F[6], sq);
            // let z = Math.atan2(tvec.data64F[4], tvec.data64F[0]);
            // console.log("Rotation (x, y, z): ", x, y, z);

            // console.log("Rotation (x, y, z) degree: ", (x / Math.PI) * 180, (y / Math.PI) * 180, (z / Math.PI) * 180);
        }
    } catch (err) {
        console.log(err);
    }
}


//! [Match blob from canvas]
async function matchBlob() {
    try {
        $("#progress").show();

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

        isRecognizing = true;
        await recognizeFace(formData);
        isRecognizing = false;
        $("#progress").hide();
    } finally {
        isEnrolling = false;
        $("#progressEnroll").hide();
    }
}
//! [Match blob from canvas]



//! [add event listener on selected image from client machine to match]
function addImageEventListeners() {
    matchInputElement.addEventListener('change', (e) => {
        matchimageElement.src = URL.createObjectURL(e.target.files[0]);
        console.log("Source changes....");
    }, false);

    matchimageElement.onload = function () {
        let frame = cv.imread(matchimageElement);
        let frameBGR = new cv.Mat(frame.cols, frame.rows, cv.CV_8UC3);
        cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

        // Process frame for detection
        processFrame(frame, frameBGR);
    };
}
//! [add image from client machine and show to canvas]

//! [match image from client machine]
async function matchImage() {
    let name = $("#name").val();
    if (name == null || name == undefined || name === "") {
        showMessage("Please enter a valid name...");
        return;
    }

    // processFrame(frame, frameBGR);
    await matchBlob();
}
//! [match image from client machine]

//! [enroll webcam from client webcam]
async function onMatchFromWebcam() {
    let name = $("#name").val();
    if (name == null || name == undefined || name === "") {
        showMessage("Please enter a valid name...");
        return;
    }
    processWebcamAsync(playWebcam);
}
//! [enroll photo from client webcam]


async function playWebcam() {
    isRunning = true;
    $("#startStopButton").html(webCamInactive);
    captureFrame();
}
