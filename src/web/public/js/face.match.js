let persons = {};
let isRunning = false;
let isRecognizing = false;

const webCamActive = "Webcam";
const webCamInactive = "Webcam Off";

let output = undefined; // document.getElementById('output');
let faceCanvas = undefined; // document.getElementById("face");
let matchimageElement = undefined;
let matchInputElement = undefined;

let toast = undefined;

//! [On document ready]
function onDocumentReady() {
    $(document).ready(function () {
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
    });
}
//! [document ready]

//! [opencv ready]
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

                if(matchScore < 0) {
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

        detectFaceMask(frameBGR);

        return;

        var faces = detectFaces(frameBGR);
        if (faces.length <= 0) {
            console.log("No face detected!");
            return false;
        }

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

//! [Match blob from canvas]
async function matchBlob() {
    try {
        $("#progress").show();

        // get name from input
        let name = $("#name").val();
        // get blob from canvas
        let blob = await getCanvasBlob(faceCanvas);

        // if no blob data found then return
        if(blob === null || blob === undefined) {
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
