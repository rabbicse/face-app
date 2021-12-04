let output = undefined; // document.getElementById('output');
let faceCanvas = undefined; // document.getElementById("face");
let matchimageElement = undefined;
let matchInputElement = undefined;
let canvas = undefined;
let faceMesh = undefined;
let toast = undefined;
const radius = 3;

//! [On document ready]
function onDocumentReady() {
    $(document).ready(function () {
        //! [initialize variables based on dom]
        matchInputElement = document.getElementById('browse');
        matchimageElement = document.getElementById('imgSrc');
        canvas = document.getElementById('output');
        //! [initialize variables based on dom]

        // Show hide
        $("#progress").hide();

        // Toast
        var toastElList = [].slice.call(document.querySelectorAll('.toast'))
        var toastList = toastElList.map(function (toastEl) {
            return new bootstrap.Toast(toastEl)
        });
        toast = toastList[0];

        //! [Add image event listeners]
        addImageEventListeners();
        //! [Add image event listeners]
    });
}
//! [document ready]

//! [opencv ready]
async function onOpenCvReady() {
    output = document.getElementById('output');
    faceCanvas = document.getElementById("face");
    console.log("opencv is ready...");

    // load dnn models
    // loadModels(processAsync);
    // await initializeDnn();

    showMessage("Dnn models loaded...");

    faceMesh = new FaceMesh({
        locateFile: (file) => {
            console.log(`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`);
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
        }
    });
    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    faceMesh.onResults(onResults);

    showMessage("Face mesh initialized!");
}
//! [opencv ready]

function showMessage(msg) {
    $('#status').html(msg);
    toast.show();
}


//! [add event listener on selected image from client machine to match]
function addImageEventListeners() {
    matchInputElement.addEventListener('change', (e) => {
        matchimageElement.src = URL.createObjectURL(e.target.files[0]);
        console.log("Source changes....");
    }, false);

    matchimageElement.onload = function () {
        // let frame = cv.imread(matchimageElement);
        // let frameBGR = new cv.Mat(frame.cols, frame.rows, cv.CV_8UC3);
        // cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

        // Process frame for detection
        // processFrame(frame, frameBGR);

        faceMesh.send({ image: matchimageElement });
    };
}
//! [add image from client machine and show to canvas]


function onResults(results) {
    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {

            // console.log(landmarks);
            // console.log(landmarks[0]);

            let nose = landmarks[1];
            let leftEye = landmarks[468]; // landmarks[130]; // stable
            let rightEye = landmarks[473] // landmarks[359];// ;
            let leftLip = landmarks[78];
            let rightLip = landmarks[308];
            let chick = landmarks[152];


            let lm = {
                nose: { x: nose.x, y: nose.y },
                chin: { x: chick.x, y: chick.y },
                leftEye: { x: leftEye.x, y: leftEye.y },
                rightEye: { x: rightEye.x, y: rightEye.y },
                leftLip: { x: leftLip.x, y: leftLip.y },
                rightLip: { x: rightLip.x, y: rightLip.y }
            }

            // let inp = document.getElementById('output');

            let im = cv.imread(matchimageElement);

            cv.circle(im, { x: nose.x * im.cols, y: nose.y * im.rows }, radius, [0, 255, 0, 255], -1); // nose
            cv.circle(im, { x: chick.x * im.cols, y: chick.y * im.rows }, radius, [0, 255, 0, 255], -1); // chin
            cv.circle(im, { x: leftEye.x * im.cols, y: leftEye.y * im.rows }, radius, [0, 0, 255, 255], -1);
            cv.circle(im, { x: rightEye.x * im.cols, y: rightEye.y * im.rows }, radius, [0, 255, 0, 255], -1);
            cv.circle(im, { x: leftLip.x * im.cols, y: leftLip.y * im.rows }, radius, [0, 0, 255, 255], -1);
            cv.circle(im, { x: rightLip.x * im.cols, y: rightLip.y * im.rows }, radius, [0, 255, 0, 255], -1);

            estimatePose(lm, im, canvas);

            cv.imshow(canvas, im);
        }
    }
}


function estimatePose(landmarks, im, canvas) {

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


function drawCircle(context, point, width, height, radius) {
    context.beginPath();
    context.arc(point.x * width, point.y * height, radius, 0, 2 * Math.PI, false);
    context.fillStyle = 'green';
    context.fill();
}