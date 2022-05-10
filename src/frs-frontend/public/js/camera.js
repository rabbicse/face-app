const FPS = 15;  // Target number of frames processed per second.
let camera = undefined;

//! [Play webcam using userMedia]
function processWebcamAsync(camera) {
    return new Promise(async function (resolve, reject) {
        try {
            // Initialize variables to set track width and height
            let cameraFrameWidth = undefined;
            let cameraFrameHeight = undefined;


            let cameraConstraints = {
                video: { facingMode: "user" },
                audio: false
            };

            // Get a permission from user to use a camera.
            let stream = await navigator.mediaDevices.getUserMedia(cameraConstraints);

            // get track info and get all camera resolution
            let cameras = []
            stream.getTracks().forEach(function (track) {
                // get track settings
                let settings = track.getSettings();

                // append width and height of usher facing camera
                cameras.push({ width: settings["width"], height: settings["height"] });
                return;
            });

            if (cameras.length == 0) {
                showMessage(
                    "No workable camera devices found!");
                return;
            }

            // set camera width and height
            cameraFrameWidth = cameras[0].width;
            cameraFrameHeight = cameras[0].height;

            // set stream to camera object
            camera.setAttribute("width", cameraFrameWidth);
            camera.setAttribute("height", cameraFrameHeight);
            console.log("camera width: ", camera.width, " camera height: ", camera.height);
            camera.srcObject = stream;

            camera.onloadedmetadata = function (e) {
                // start camera to stream frames from webcam 
                // camera.play();

                console.log("video played...");

                // call callback function to process frame
                // setTimeout(callback, 1);

                // console.log("called callback...");

                resolve(true);
            };
        } catch (error) {
            console.log("Error when initialize camera. Details: ", error);            
            reject(false);
        }
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