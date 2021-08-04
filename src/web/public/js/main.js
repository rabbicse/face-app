let netDet = undefined;
let netRecogn = undefined;
let persons = {};
let camera = undefined;
let isRunning = false;
const FPS = 30;  // Target number of frames processed per second.
let output = undefined; // document.getElementById('output');
let faceCanvas = undefined; // document.getElementById("face");

function onOpenCvReady() {
    document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
    console.log("opencv is ready...");

    output = document.getElementById('output');
    faceCanvas = document.getElementById("face");

    // Create a camera object.
    camera = document.createElement("video");
    camera.setAttribute("width", output.width);
    camera.setAttribute("height", output.height);
    console.log(camera);

    // load dnn models
    loadModels(processAsync);
}

function loadModels(callback) {
    var utils = new Utils('');
    var proto = '/models/deploy_lowres.prototxt';
    var weights = '/models/res10_300x300_ssd_iter_140000_fp16.caffemodel';
    var recognModel = 'https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7';
    utils.createFileFromUrl('face_detector.prototxt', proto, () => {
        document.getElementById('status').innerHTML = 'Downloading face_detector.caffemodel';
        utils.createFileFromUrl('face_detector.caffemodel', weights, () => {
            document.getElementById('status').innerHTML = 'Downloaded face_detector.caffemodel';
            netDet = cv.readNetFromCaffe('face_detector.prototxt', 'face_detector.caffemodel');

            callback()
        });
    });
};

function processAsync() {
    console.log("dnn models loaded...");

    // Get a permission from user to use a camera.
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function (stream) {
            camera.srcObject = stream;
            camera.onloadedmetadata = function (e) {
                camera.play();
            };
        });

    document.getElementById('startStopButton').disabled = false;
}

//! [Run face detection model]
function detectFaces(img) {
    var blob = cv.blobFromImage(img, 1, { width: 192, height: 144 }, [104, 117, 123, 0], false, false);
    netDet.setInput(blob);
    var out = netDet.forward();

    var faces = [];
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
    blob.delete();
    out.delete();
    return faces;
};
//! [Run face detection model]

//! [Recognize face]
function recognizeFace(formData) {
    try {
        $.ajax({
            method: "POST",
            url: "http://localhost:5000/match/v1",
            data: formData,
            processData: false,
            contentType: false,
            timeout: 0,
            mimeType: "multipart/form-data",
            success: function (result) {
                console.log(result);
                var parsedJson = $.parseJSON(result);
                $("#score").html(parsedJson["score"]);
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
function captureFrame() {
    //! [Open a camera stream]
    var cap = new cv.VideoCapture(camera);
    var frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
    var frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);
    //! [Open a camera stream]

    cap.read(frame);  // Read a frame from camera
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    var begin = Date.now();

    processFrame(frame, frameBGR);

    cv.imshow(output, frame);

    // Loop this function.
    if (isRunning) {
        var delay = 1000 / FPS - (Date.now() - begin);
        setTimeout(captureFrame, delay);
    }


};
//! [Define frames processing]

//!
function processFrame(frame, frameBGR) {
    var faces = detectFaces(frameBGR);
    faces.forEach(function (rect) {
        // cv.rectangle(frame, { x: rect.x, y: rect.y }, { x: rect.x + rect.width, y: rect.y + rect.height }, [0, 255, 0, 255]);

        var face = frame.roi(rect);
        // face = cv.cvtColor(face, cv.COLOR_RGBA2BGR);
        cv.imshow(faceCanvas, face);

        // let mat = new cv.Mat();
        // cv.imencode("jpg", face, mat);

        // var name = recognize(face);
        // cv.putText(frame, name, { x: rect.x, y: rect.y }, cv.FONT_HERSHEY_SIMPLEX, 1.0, [0, 255, 0, 255]);

        // recognizeFace(face);

        faceCanvas.toBlob(function (blob) {
            const formData = new FormData();
            formData.append("image", blob, "photo.jpg");
            formData.append("data", JSON.stringify({ "name": "rabbi" }));
            recognizeFace(formData);
        }, "image/jpeg", 0.95);
    });
}

//! [add image from client machine and show to canvas]
function loadImage() {
    let imgElement = document.getElementById('matchImageSrc');
    let inputElement = document.getElementById('matchImage');
    inputElement.addEventListener('change', (e) => {
        imgElement.src = URL.createObjectURL(e.target.files[0]);
        console.log("Source changes....");
    }, false);

    imgElement.onload = function () {
        let mat = cv.imread(imgElement);
        cv.imshow(output, mat);
        mat.delete();
    };
}
//! [add image from client machine and show to canvas]

//! [match image from client machine]
function checkImage() {
    let imgElement = document.getElementById('matchImageSrc');
    let frame = cv.imread(imgElement);
    let frameBGR = new cv.Mat(frame.cols, frame.rows, cv.CV_8UC3);
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    processFrame(frame, frameBGR);

    frame.delete();
    frameBGR.delete();
}
//! [match image from client machine]

function run() {
    isRunning = true;
    captureFrame();
    $("#startStopButton").innerHTML = 'Stop';
    $("#startStopButton").disabled = false;
}
