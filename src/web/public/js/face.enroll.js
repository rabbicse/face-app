let persons = {};
let isRunning = false;
let isRecognizing = false;
let isEnrolling = false;
let isEnrolled = false;

let output = undefined; // document.getElementById('output');
let faceCanvas = undefined; // document.getElementById("face");
let enrollImageElement = undefined;
let enrollInputElement = undefined;

let toast = undefined;

function onDocumentReady() {
    $(document).ready(async function () {

        // Update elements
        output = document.getElementById('output');
        camera_output = document.getElementById('camera_output');
        faceCanvas = document.getElementById("face");
        console.log("opencv is ready...");

        // Create a camera object.
        camera = document.createElement("video");
        // update elements

        //! [initialize variables based on dom]
        enrollInputElement = document.getElementById('enrollImage');
        enrollImageElement = document.getElementById('enrollImageSrc');
        //! [initialize variables based on dom]


        // Toast
        var toastElList = [].slice.call(document.querySelectorAll('.toast'))
        var toastList = toastElList.map(function (toastEl) {
            return new bootstrap.Toast(toastEl)
        });
        // console.log(toastList);
        // toastList.forEach(toast => toast.show()); // This show them
        toast = toastList[0];

        //! [Add image event listeners]
        addImageEventListeners();
        //! [Add image event listeners]

        //! [enroll image button event]
        $("#enrollImageButton").click(function (e) {
            e.preventDefault();
            onEnrollImage();
        });
        //! [enroll image button event]

        //! [enroll webcam button event]
        $("#enrollWebcamButton").click(function (e) {
            e.preventDefault();
            onEnrollWebcam();
        });
        //! [enroll webcam button event]


        // load opencv if exists inside indexeddb
        await initializeOpenCV();

        showMessage("Dnn models loaded...");

        // Show hide
        $("#progressEnroll").hide();
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
    await initializeDnn();

    showMessage("Dnn models loaded...");
}

function showMessage(msg) {
    $('#status').html(msg);
    toast.show();
}


//! [Enroll face]
async function enrollFace(formData) {
    try {
        console.log("ok...enrollFace");
        return $.ajax({
            method: "POST",
            url: "https://" + hostname + ":5000/enroll/v1",
            data: formData,
            processData: false,
            contentType: false,
            timeout: 0,
            mimeType: "multipart/form-data",
            success: function (result) {
                console.log("Enroll success...");
                showMessage("Enroll Success!");
                isEnrolled = true;
            },
            error: function (err) {
                console.log("Error...");
                console.log(err);
                showMessage("Enroll Error! Something went wrong!");
            }
        });
    } catch (e) {
        console.log("Error...");
        console.log(e);
        showMessage("Enroll Error! Something went wrong!");
    }
}
//! [Enroll face]


//! [Process Image to Enroll]
async function processImageToEnroll(frame, frameBGR) {
    try {
        // detect faces based on DNN
        var faces = detectFaces(frameBGR);
        // If no face detected then return
        if (faces.length <= 0) {
            return false;
        }

        let oFrame = frame.clone();
        faces.forEach(function (rect) {
            cv.rectangle(frame,
                { x: rect.x, y: rect.y },
                { x: rect.x + rect.width, y: rect.y + rect.height },
                [0, 255, 0, 255],
                2);
        });

        console.log("showing frame...");
        cv.imshow(output, frame);

        // If more than 1 face detected then return
        if (faces.length !== 1) {
            return false;
        }

        // Get detected face and estimate rect to crop
        let rect = faces[0];
        let xTh = rect.width / 3;
        let yTh = rect.height / 3;
        let x = Math.max(rect.x - xTh, 0);
        let y = Math.max(rect.y - yTh, 0);
        let x1 = Math.min(rect.x + rect.width + xTh, frame.cols);
        let y1 = Math.min(rect.y + rect.height + yTh, frame.rows);
        var rct = new cv.Rect(x, y, x1 - x, y1 - y);

        // Crop detected face
        var face = oFrame.roi(rct);
        cv.imshow(faceCanvas, face);
        return true;
    } catch (err) {
        console.log(err);
    } finally {
        frame.delete();
        frameBGR.delete();
    }
    return false;
}
//! [Process Image to Enroll]

//! [Enroll blob from canvas]
async function enrollBlob() {
    try {
        $("#progressEnroll").show();
        let blob = await getCanvasBlob(faceCanvas);
        isEnrolling = true;
        // Prepare form data to call enroll service
        const formData = new FormData();
        formData.append("image", blob, "photo.jpg");
        formData.append("data", JSON.stringify({ "name": $("#enrollName").val() }));
        console.log(formData);
        await enrollFace(formData);
    } finally {
        isEnrolling = false;
        $("#progressEnroll").hide();
    }
}
//! [Process Image to Enroll]


//! [add event listener on selected image from client machine to match]
function addImageEventListeners() {
    enrollInputElement.addEventListener('change', (e) => {
        enrollImageElement.src = URL.createObjectURL(e.target.files[0]);
        console.log("Source changes....");
    }, false);

    enrollImageElement.onload = function () {
        let frame = cv.imread(enrollImageElement);
        // cv.imshow(output, mat);
        // mat.delete();

        // let frame = cv.imread(enrollImageElement);
        let frameBGR = new cv.Mat(frame.cols, frame.rows, cv.CV_8UC3);
        cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

        // Process frame for detection
        processImageToEnroll(frame, frameBGR);
    };
}
//! [add image from client machine and show to canvas]


//! [enroll image from client machine]
async function onEnrollImage() {
    await enrollBlob();
}
//! [enroll image from client machine]


//! [enroll webcam from client webcam]
async function onEnrollWebcam() {
    let name = $("#enrollName").val();
    if (name == null || name == undefined || name === "") {
        showMessage("Please enter a valid name...");
        return;
    }
    processWebcamAsync(enrollFrame);

}
//! [enroll photo from client webcam]

//! [enroll frame from Webcam]
async function enrollFrame() {
    //! [Open a camera stream]
    var cap = new cv.VideoCapture(camera);
    var frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
    var frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);
    //! [Open a camera stream]

    // Read a frame from camera
    cap.read(frame);
    // Convert from RGBA to BGR
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    var begin = Date.now();

    // If image processed & face detected
    if (await processImageToEnroll(frame, frameBGR)) {
        // Now preprocess & call service etc.
        await enrollBlob();
    }

    // Loop this function.
    if (!isEnrolled) {
        console.log("not enrolled.....");
        var delay = 1000 / FPS - (Date.now() - begin);
        setTimeout(enrollFrame, delay);
    } else {
        console.log("enrolled...");
        isEnrolled = false;
        stopStreamedVideo(camera);
    }
}
//