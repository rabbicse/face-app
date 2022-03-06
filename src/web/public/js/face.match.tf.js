let persons = {};
let isRunning = false;
let isRecognizing = false;

const webCamActive = "Webcam";
const webCamInactive = "Webcam Off";

let output = undefined; // document.getElementById('output');
let faceCanvas = undefined; // document.getElementById("face");
let faceCanvasHidden = undefined; // document.getElementById("face");
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

        // Init toast
        toast = initToast();

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
    camera_output = document.getElementById('camera_output');
    output = document.getElementById('output');
    faceCanvas = document.getElementById("face");
    faceCanvasHidden = document.getElementById("face_hidden");
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



//! [Define frames processing]
async function captureFrame() {
    //! [Open a camera stream]
    var cap = new cv.VideoCapture(camera);
    console.log("Camera height: ", camera.height, " Camera width: ", camera.width)
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

//! [Process frame]
async function processFrame(frame, frameBGR) {
    try {
        // detect faces if no faces then return
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
        let xTh = rect.width / 3;
        let yTh = rect.height / 3;
        let x = Math.max(rect.x - xTh, 0);
        let y = Math.max(rect.y - yTh, 0);
        let x1 = Math.min(rect.x + rect.width + xTh, frame.cols);
        let y1 = Math.min(rect.y + rect.height + yTh, frame.rows);
        let rct = new cv.Rect(x, y, x1 - x, y1 - y);
        // console.log(rct);
        let face = oFrame.roi(rct);
        // face = cv.cvtColor(face, cv.COLOR_RGBA2BGR);
        cv.imshow(faceCanvas, face);
        cv.imshow(faceCanvasHidden, face);

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

        console.log("Face Size: ", faceCanvas.width, " x ", faceCanvas.height);
        console.log("Face Size hidden: ", faceCanvasHidden.width, " x ", faceCanvasHidden.height);

        // get blob from canvas
        let blob = await getCanvasBlob(faceCanvasHidden);

        // if no blob data found then return
        if (blob === null || blob === undefined) {
            showMessage("No detected frame found!");
            return;
        }

        // create form data object to call service
        const formData = new FormData();
        formData.append("image", blob, "photo.jpg");
        formData.append("data", JSON.stringify({ "name": name }));


        try {
            isRecognizing = true;
            await recognizeFace(formData);
        } catch (err) {
            console.log(err);

        } finally {
            isRecognizing = false;
        }
    } catch (exp) {
        console.log(exp);
    } finally {
        isEnrolling = false;
        $("#progressEnroll").hide();
        $("#progress").hide();
    }
}
//! [Match blob from canvas]



// //! [add event listener on selected image from client machine to match]
// function addImageEventListeners() {
//     matchInputElement.addEventListener('change', (e) => {
//         matchimageElement.src = URL.createObjectURL(e.target.files[0]);
//         console.log("Source changes....");
//     }, false);

//     matchimageElement.onload = function () {
//         let frame = cv.imread(matchimageElement);
//         let frameBGR = new cv.Mat(frame.cols, frame.rows, cv.CV_8UC3);
//         cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

//         // Process frame for detection
//         processFrame(frame, frameBGR);
//     };
// }
// //! [add image from client machine and show to canvas]


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
        // processFrame(frame, frameBGR);

        detectMask(matchimageElement).then((results) => {
            let hasMask = false;
            for (bboxInfo of results) {
                bbox = bboxInfo[0];
                classID = bboxInfo[1];
                score = bboxInfo[2];
                console.log("class ID: " + classID);

                if (classID === 0) {
                    hasMask = true;
                    break;
                }
            }

            if (hasMask) {
                showMessage("Face mask detected!!!");
                return;
            }

            // Process frame for detection
            processFrame(frame, frameBGR);
        });
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
