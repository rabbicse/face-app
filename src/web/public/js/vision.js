const protoDownloadPath = '/models/deploy_lowres.prototxt';
const weightsDownloadPath = '/models/res10_300x300_ssd_iter_140000_fp16.caffemodel';
const protoPath = 'face_detector.prototxt';
const weightsPath = 'face_detector.caffemodel';
const FPS = 15;  // Target number of frames processed per second.
let netDet = undefined;
let netRecogn = undefined;
let camera = undefined;

//! [Run face detection model]
function detectFaces(img) {
    var faces = [];

    try {
        var blob = cv.blobFromImage(img, 1, { width: 192, height: 144 }, [104, 117, 123, 0], false, false);
        netDet.setInput(blob);
        var out = netDet.forward();


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
            showMessage("No face detecteed...");
        }
        blob.delete();
        out.delete();
    } catch (ex) {
        console.log("Error when apply face detection");
    }
    return faces;
};
//! [Run face detection model]

//! [Initialize DNN]
async function initializeDnn() {
    try {
        // Download proto file for caffe model
        await downloadFileAsync(protoPath, protoDownloadPath);
        console.log("proto downloaded...");

        await downloadFileAsync('face_detector.caffemodel', weightsDownloadPath);

        console.log("caffemodel downloaded...");

        await sleep(500);

        netDet = cv.readNetFromCaffe('face_detector.prototxt', 'face_detector.caffemodel');

        console.log("net loaded...");
    } catch (err) {
        console.log(err);
    }
}
//! [Initialize DNN]

//! [Download file from http to browser path]
async function downloadFileAsync(path, uri) {
    try {
        return $.ajax({
            method: "GET",
            xhrFields: {
                responseType: 'arraybuffer'
            },
            url: uri,
            timeout: 0,
            success: function (response) {
                console.log("download success...");
                let data = new Uint8Array(response);
                cv.FS_createDataFile('/', path, data, true, false, false);
            },
            error: function (err) {
                console.log(err);
            }
        });
    } catch (ex) {
        console.log(ex);
    }
}
//! [Download file from http to browser path]

//! [Play webcam using userMedia]
function processWebcamAsync(callback) {
    // Get a permission from user to use a camera.
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function (stream) {
            camera.srcObject = stream;
            camera.onloadedmetadata = function (e) {
                camera.play();

                // callback
                callback();
            };
        });
}
//! [Play webcam using userMedia]

//! [Stop all user media]
function stopStreamedVideo(videoElem) {
    const stream = videoElem.srcObject;
    const tracks = stream.getTracks();

    tracks.forEach(function (track) {
        track.stop();
    });

    videoElem.srcObject = null;
}
//! [Stop all user media]

//! [Convert Canvas to blob]
function getCanvasBlob(canvas) {
    return new Promise(function (resolve, reject) {
        canvas.toBlob(function (blob) {
            resolve(blob)
        }, "image/jpeg", 0.95)
    })
}
//! [Convert Canvas to blob]

//! [Thread Sleep]
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
//! [Thread Sleep]