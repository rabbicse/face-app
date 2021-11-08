// Based on python code provided in "Head Pose Estimation using OpenCV and Dlib"
//   https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/#code
// 3D model points


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
        const size = { width: 640, height: 480 };
        const focalLength = size.width;
        const center = [size.width / 2, size.height / 2];
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
        let pointZ = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 0.0, 500.0]);
        let pointY = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 500.0, 0.0]);
        let pointX = cv.matFromArray(1, 3, cv.CV_64FC1, [500.0, 0.0, 0.0]);
        let noseEndPoint2DZ = new cv.Mat();
        let nose_end_point2DY = new cv.Mat();
        let nose_end_point2DX = new cv.Mat();
        let jaco = new cv.Mat();
        let R = new cv.Mat();
        let T = new cv.Mat();


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
            true
        );

        // console.log(success);

        if (success) {
            // console.log("Rotation Vector:", rvec.data64F);
            // console.log(
            //     "Rotation Vector (in degree):",
            //     rvec.data64F.map(d => (d / Math.PI) * 180));


            // Project a 3D points [0.0, 0.0, 500.0],  [0.0, 500.0, 0.0],
            //   [500.0, 0.0, 0.0] as z, y, x axis in red, green, blue color


            // console.log(cv.projectPoints);

            cv.projectPoints(
                pointZ,
                rvec,
                tvec,
                cameraMatrix,
                distCoeffs,
                noseEndPoint2DZ,
                jaco
            );


            cv.Rodrigues(rvec, R);

            console.log("R", R.rows, R.cols);

            console.log("TVEC", tvec.rows, tvec.cols);

            // Initialise a MatVector
            let matVec = new cv.MatVector();
            // Push a Mat back into MatVector
            matVec.push_back(R);

            console.log(matVec.rows, matVec.cols);

            T = cv.hconcat(R, tvec);

            console.log("T: ", T.rows, T.cols);

            let euler_angle = new cv.Mat();
            let out_rotation = new cv.Mat();
            let out_translation = new cv.Mat();

            let x = new cv.Mat();
            let y = new cv.Mat();
            let z = new cv.Mat();

            cv.decomposeProjectionMatrix(cameraMatrix,
                    R, matVec, x, y, z, euler_angle);

            console.log("Euler angle: ", euler_angle);


            // // color the detected eyes and nose to purple
            // for (var i = 0; i < numRows; i++) {
            //     cv.circle(
            //         im,
            //         {
            //             x: imagePoints.doublePtr(i, 0)[0],
            //             y: imagePoints.doublePtr(i, 1)[0]
            //         },
            //         3,
            //         [255, 0, 255, 255],
            //         -1
            //     );
            // }

            // // cv.imshow(canvas, im);
        }
    } catch (err) {
        console.log(err);
    }

}

function main(sources) {
    // Create Matrixes
    const imagePoints = cv.Mat.zeros(numRows, 2, cv.CV_64FC1);
    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
    const rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
    const tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
    const pointZ = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 0.0, 500.0]);
    const pointY = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 500.0, 0.0]);
    const pointX = cv.matFromArray(1, 3, cv.CV_64FC1, [500.0, 0.0, 0.0]);
    const noseEndPoint2DZ = new cv.Mat();
    const nose_end_point2DY = new cv.Mat();
    const nose_end_point2DX = new cv.Mat();
    const jaco = new cv.Mat();
    window.beforeunload = () => {
        im.delete();
        imagePoints.delete();
        distCoeffs.delete();
        rvec.delete();
        tvec.delete();
        pointZ.delete();
        pointY.delete();
        pointX.delete();
        noseEndPoint2DZ.delete();
        nose_end_point2DY.delete();
        nose_end_point2DX.delete();
        jaco.delete();
    };

    // main event loop
    sources.PoseDetection.poses.addListener({
        next: poses => {
            // skip if no person or more than one person is found
            if (poses.length !== 1) {
                return;
            }

            const person1 = poses[0];
            if (
                !person1.keypoints.find(kpt => kpt.part === "nose") ||
                !person1.keypoints.find(kpt => kpt.part === "leftEye") ||
                !person1.keypoints.find(kpt => kpt.part === "rightEye")
            ) {
                return;
            }
            const ns = person1.keypoints.filter(kpt => kpt.part === "nose")[0]
                .position;
            const le = person1.keypoints.filter(kpt => kpt.part === "leftEye")[0]
                .position;
            const re = person1.keypoints.filter(kpt => kpt.part === "rightEye")[0]
                .position;

            // 2D image points. If you change the image, you need to change vector
            [
                ns.x,
                ns.y, // Nose tip
                ns.x,
                ns.y, // Nose tip (see HACK! above)
                // 399, 561, // Chin
                le.x,
                le.y, // Left eye left corner
                re.x,
                re.y // Right eye right corner
                // 345, 465, // Left Mouth corner
                // 453, 469 // Right mouth corner
            ].map((v, i) => {
                imagePoints.data64F[i] = v;
            });

            // Hack! initialize transition and rotation matrixes to improve estimation
            tvec.data64F[0] = -100;
            tvec.data64F[1] = 100;
            tvec.data64F[2] = 1000;
            const distToLeftEyeX = Math.abs(le.x - ns.x);
            const distToRightEyeX = Math.abs(re.x - ns.x);
            if (distToLeftEyeX < distToRightEyeX) {
                // looking at left
                rvec.data64F[0] = -1.0;
                rvec.data64F[1] = -0.75;
                rvec.data64F[2] = -3.0;
            } else {
                // looking at right
                rvec.data64F[0] = 1.0;
                rvec.data64F[1] = -0.75;
                rvec.data64F[2] = -3.0;
            }

            const success = cv.solvePnP(
                modelPoints,
                imagePoints,
                cameraMatrix,
                distCoeffs,
                rvec,
                tvec,
                true
            );
            if (!success) {
                return;
            }
            console.log("Rotation Vector:", rvec.data64F);
            console.log(
                "Rotation Vector (in degree):",
                rvec.data64F.map(d => (d / Math.PI) * 180)
            );
            console.log("Translation Vector:", tvec.data64F);

            // Project a 3D points [0.0, 0.0, 500.0],  [0.0, 500.0, 0.0],
            //   [500.0, 0.0, 0.0] as z, y, x axis in red, green, blue color
            cv.projectPoints(
                pointZ,
                rvec,
                tvec,
                cameraMatrix,
                distCoeffs,
                noseEndPoint2DZ,
                jaco
            );
            cv.projectPoints(
                pointY,
                rvec,
                tvec,
                cameraMatrix,
                distCoeffs,
                nose_end_point2DY,
                jaco
            );
            cv.projectPoints(
                pointX,
                rvec,
                tvec,
                cameraMatrix,
                distCoeffs,
                nose_end_point2DX,
                jaco
            );

            let im = cv.imread(document.querySelector("canvas"));
            // color the detected eyes and nose to purple
            for (var i = 0; i < numRows; i++) {
                cv.circle(
                    im,
                    {
                        x: imagePoints.doublePtr(i, 0)[0],
                        y: imagePoints.doublePtr(i, 1)[0]
                    },
                    3,
                    [255, 0, 255, 255],
                    -1
                );
            }
            // draw axis
            const pNose = { x: imagePoints.data64F[0], y: imagePoints.data64F[1] };
            const pZ = {
                x: noseEndPoint2DZ.data64F[0],
                y: noseEndPoint2DZ.data64F[1]
            };
            const p3 = {
                x: nose_end_point2DY.data64F[0],
                y: nose_end_point2DY.data64F[1]
            };
            const p4 = {
                x: nose_end_point2DX.data64F[0],
                y: nose_end_point2DX.data64F[1]
            };
            cv.line(im, pNose, pZ, [255, 0, 0, 255], 2);
            cv.line(im, pNose, p3, [0, 255, 0, 255], 2);
            cv.line(im, pNose, p4, [0, 0, 255, 255], 2);

            // Display image
            cv.imshow(document.querySelector("canvas"), im);
            im.delete();
        }
    });

    const params$ = xs.of({
        singlePoseDetection: { minPoseConfidence: 0.2 }
    });
    const vdom$ = sources.PoseDetection.DOM;

    return {
        DOM: vdom$,
        PoseDetection: params$
    };
}