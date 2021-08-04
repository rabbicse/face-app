let netDet = undefined;
let netRecogn = undefined;
let persons = {};
let camera = undefined;
let isRunning = false;
let isRecognizing = false;

const webCamActive = "Webcam";
const webCamInactive = "Webcam Off";
const FPS = 15;  // Target number of frames processed per second.
const proto = '/models/deploy_lowres.prototxt';
const weights = '/models/res10_300x300_ssd_iter_140000_fp16.caffemodel';

let output = undefined; // document.getElementById('output');
let faceCanvas = undefined; // document.getElementById("face");
let matchimageElement = undefined;
let matchInputElement = undefined;
let enrollImageElement = undefined;
let enrollInputElement = undefined;

function onDocumentReady() {
    $(document).ready(function () {
        //! [initialize variables based on dom]
        matchInputElement = document.getElementById('matchImage');
        matchimageElement = document.getElementById('matchImageSrc');

        enrollInputElement = document.getElementById('enrollImage');
        enrollImageElement = document.getElementById('enrollImageSrc');
        //! [initialize variables based on dom]

        // Show hide
        $("#progress").hide();
        $("#progressEnroll").hide();

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
            } else {
                // Get a permission from user to use a camera.
                navigator.mediaDevices.getUserMedia({ video: true, audio: false })
                    .then(function (stream) {
                        camera.srcObject = stream;
                        camera.onloadedmetadata = function (e) {
                            camera.play();

                            playWebcam();
                        };
                    });
            }
        });
        //! [add camera start/stop button event]

        //! [enroll image button event]
        $("#enrollImageButton").click(function (e) {
            e.preventDefault();
            onEnrollImage();
        });
        //! [enroll image button event]

        //! [match image button event]
        $("#matchImageButton").click(function (e) {
            e.preventDefault();
            checkImage();
        });
    });
}

async function onOpenCvReady() {
    output = document.getElementById('output');
    faceCanvas = document.getElementById("face");
    console.log("opencv is ready...");

    // Create a camera object.
    camera = document.createElement("video");
    camera.setAttribute("width", output.width);
    camera.setAttribute("height", output.height);

    // load dnn models
    // loadModels(processAsync);
    await initDnn();

    $('#status').html('Loaded...');
}

async function initDnn() {
    try {
        await downloadFileAsync('face_detector.prototxt', proto);

        console.log("proto downloaded...");

        await downloadFileAsync('face_detector.caffemodel', weights);

        console.log("caffemodel downloaded...");

        await sleep(500);

        netDet = cv.readNetFromCaffe('face_detector.prototxt', 'face_detector.caffemodel');

        console.log("net loaded...");
    } catch (err) {
        console.log(err);
    }
}

function checkIfFileLoaded(fileName) {
    $.get(fileName, function (data, textStatus) {
        if (textStatus == "success") {
            // execute a success code
            console.log("file loaded!");
        }
    });
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function downloadFileAsync(path, uri) {
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
        error: function () {

        }
    });
}

function processAsync() {
    // Get a permission from user to use a camera.
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function (stream) {
            camera.srcObject = stream;
            camera.onloadedmetadata = function (e) {
                camera.play();
            };
        });
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
async function enrollFace(formData) {
    try {
        console.log("ok...enrollFace");
        return $.ajax({
            method: "POST",
            url: "http://localhost:5000/enroll/v1",
            data: formData,
            processData: false,
            contentType: false,
            timeout: 0,
            mimeType: "multipart/form-data",
            success: function (result) {
                console.log("Enroll success...");
                // console.log(result);
            },
            error: function (err) {
                console.log("Error...");
                console.log(err);
            }
        });
    } catch (e) {
        console.log("Error...");
        console.log(e);
    }
}
//! [Recognize face]

//! [Recognize face]
async function recognizeFace(formData) {
    try {
        return $.ajax({
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
                var matchScore = parsedJson["score"];
                var score = (parseFloat(matchScore) * 100).toFixed(2) + '%';
                $("#score").html(score);

                if (matchScore >= 0.60) {
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

    if (!isRecognizing) {
        await processFrame(frame, frameBGR);
    }

    cv.imshow(output, frame);

    // Loop this function.
    if (isRunning) {
        var delay = 1000 / FPS - (Date.now() - begin);
        setTimeout(captureFrame, delay);
    }
};
//! [Define frames processing]

//!
async function processFrame(frame, frameBGR) {
    var matchName = $("#name").val();

    if (matchName === '') {
        return;
    }

    var faces = detectFaces(frameBGR);
    if (faces.length !== 1) {
        return;
    }

    faces.forEach(function (rect) {
        // cv.rectangle(frame, { x: rect.x, y: rect.y }, { x: rect.x + rect.width, y: rect.y + rect.height }, [0, 255, 0, 255]);
        let xTh = rect.width / 10;
        let yTh = rect.height / 10;
        let x = Math.max(rect.x - xTh, 0);
        let y = Math.max(rect.y - yTh, 0);
        let x1 = Math.min(rect.x + rect.width + xTh, frame.cols);
        let y1 = Math.min(rect.y + rect.height + yTh, frame.rows);
        var rct = new cv.Rect(x, y, x1 - x, y1 - y);
        console.log(rct);
        var face = frame.roi(rct);
        // face = cv.cvtColor(face, cv.COLOR_RGBA2BGR);
        cv.imshow(faceCanvas, face);

        // let mat = new cv.Mat();
        // cv.imencode("jpg", face, mat);

        // var name = recognize(face);
        // cv.putText(frame, name, { x: rect.x, y: rect.y }, cv.FONT_HERSHEY_SIMPLEX, 1.0, [0, 255, 0, 255]);

        // recognizeFace(face);

        return faceCanvas.toBlob(async function (blob) {
            $("#progress").show();
            const formData = new FormData();
            formData.append("image", blob, "photo.jpg");
            formData.append("data", JSON.stringify({ "name": matchName }));

            isRecognizing = true;
            await recognizeFace(formData);
            isRecognizing = false;
            $("#progress").hide();
        }, "image/jpeg", 0.95);

        // var blob = await getCanvasBlob(faceCanvas);
        // console.log(blob);
        // const formData = new FormData();
        // formData.append("image", blob, "photo.jpg");
        // formData.append("data", JSON.stringify({ "name": "rabbi" }));
        // await recognizeFace(formData);
    });
}

async function enrollImage(frame, frameBGR) {
    try {
        var faces = detectFaces(frameBGR);
        if (faces.length !== 1) {
            return;
        }

        faces.forEach(function (rect) {
            // cv.rectangle(frame, { x: rect.x, y: rect.y }, { x: rect.x + rect.width, y: rect.y + rect.height }, [0, 255, 0, 255]);
            let xTh = rect.width / 10;
            let yTh = rect.height / 10;
            let x = Math.max(rect.x - xTh, 0);
            let y = Math.max(rect.y - yTh, 0);
            let x1 = Math.min(rect.x + rect.width + xTh, frame.cols);
            let y1 = Math.min(rect.y + rect.height + yTh, frame.rows);
            var rct = new cv.Rect(x, y, x1 - x, y1 - y);
            console.log(rct);
            // var face = frame.roi(rect);
            var face = frame.roi(rct);
            cv.imshow(faceCanvas, face);

            faceCanvas.toBlob(async function (blob) {
                $("#progressEnroll").show();
                const formData = new FormData();
                formData.append("image", blob, "photo.jpg");
                formData.append("data", JSON.stringify({ "name": $("#enrollName").val() }));
                console.log(formData);
                await enrollFace(formData);
                $("#progressEnroll").hide();
            }, "image/jpeg", 0.95);
        });
    } catch (err) {
        console.log(err);
    } finally {
        frame.delete();
        frameBGR.delete();
    }
}


function getCanvasBlob(canvas) {
    return new Promise(function (resolve, reject) {
        canvas.toBlob(function (blob) {
            resolve(blob)
        })
    }, "image/jpeg", 0.95)
}

//! [add event listener on selected image from client machine to match]
function addImageEventListeners() {
    matchInputElement.addEventListener('change', (e) => {
        matchimageElement.src = URL.createObjectURL(e.target.files[0]);
        console.log("Source changes....");
    }, false);

    matchimageElement.onload = function () {
        let mat = cv.imread(matchimageElement);
        cv.imshow(output, mat);
        mat.delete();
    };


    enrollInputElement.addEventListener('change', (e) => {
        enrollImageElement.src = URL.createObjectURL(e.target.files[0]);
        console.log("Source changes....");
    }, false);

    enrollImageElement.onload = function () {
        let mat = cv.imread(enrollImageElement);
        cv.imshow(output, mat);
        mat.delete();
    };
}
//! [add image from client machine and show to canvas]

//! [match image from client machine]
function checkImage() {
    let frame = cv.imread(matchimageElement);
    let frameBGR = new cv.Mat(frame.cols, frame.rows, cv.CV_8UC3);
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    processFrame(frame, frameBGR);

    // frame.delete();
    // frameBGR.delete();
}
//! [match image from client machine]


//! [match image from client machine]
async function onEnrollImage() {
    let frame = cv.imread(enrollImageElement);
    let frameBGR = new cv.Mat(frame.cols, frame.rows, cv.CV_8UC3);
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    console.log("ok....onEnrollImage");
    await enrollImage(frame, frameBGR);
}
//! [match image from client machine]


async function playWebcam() {
    isRunning = true;
    $("#startStopButton").html(webCamInactive);
    await captureFrame();
}
