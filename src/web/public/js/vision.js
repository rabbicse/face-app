const protoDownloadPath = '/models/deploy_lowres.prototxt';
const weightsDownloadPath = '/models/res10_300x300_ssd_iter_140000_fp16.caffemodel';

const maskProtoDownloadPath = '/models/face_mask_detection.prototxt';
const maskWeightsDownloadPath = '/models/face_mask_detection.caffemodel';

const protoPath = 'face_detector.prototxt';
const weightsPath = 'face_detector.caffemodel';

const maskMrotoPath = 'face_mask_detection.prototxt';
const maskWeightsPath = 'face_mask_detection.caffemodel';

const FPS = 15;  // Target number of frames processed per second.
let netDet = undefined;
let netRecogn = undefined;
let netMask = undefined;
let netMaskTf = undefined;
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


//! [Run face detection model]
function detectFaceMask(img) {
    var faces = [];

    try {
        // Get image height width and channels
        // height, width, _ = img.shape
        // console.log("Height: " + height + " Width: " + width);

        // var layersNames = netMask.getLayerNames();
        // console.log(layersNames);


        // Convert image to blob
        let frameRGB = new cv.Mat(img.cols, img.rows, cv.CV_8UC3);
        cv.cvtColor(img, frameRGB, cv.COLOR_BGR2RGB);
        // var blob = cv.blobFromImage(frameRGB, 1 / 255.0, { width: 260, height: 260 });
        // blob = cv2.dnn.blobFromImage(image, 1 / 255.0,  size = target_shape)
        // netMask.setInput(blob);

        // Get the names of all the layers in the network
        // layersNames = netMask.layerNames
        // console.log(layersNames)



        var blob = cv.blobFromImage(frameRGB, 1 / 255.0, { width: 260, height: 260 });
        console.log('ok.....0');
        netMask.setInput(blob, "data");
        // const out = netMask.forward();

        var output = netMask.forward();
        for (i = 0, n = output.data32F.length; i < n; i++) {
            console.log(output.data32F[i]);
        }

        // console.log(output)

        // const bboxes = decodeBBox(anchors, tf.squeeze(rawBBoxes));
        // const Results = nonMaxSuppression(bboxes, tf.squeeze(rawConfidences), 0.5, 0.5,  width, height );


        // console.log('ok....1');
        // console.log(out);
        // console.log('ok....2');


        // y_bboxes_output, y_cls_output = net.forward(getOutputsNames(net))
        // // remove the batch dimension, for batch is always 1 for inference.
        // y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
        // y_cls = y_cls_output[0]
        // // To speed up, do single class NMS, not multiple classes NMS.
        // bbox_max_scores = np.max(y_cls, axis = 1)
        // bbox_max_score_classes = np.argmax(y_cls, axis = 1)

        // // keep_idx is the alive bounding box after nms.
        // keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh = conf_thresh, iou_thresh = iou_thresh)
        // // keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
        // tl = round(0.002 * (height + width) * 0.5) + 1  // line thickness
        // for idx in keep_idxs:
        //     conf = float(bbox_max_scores[idx])
        // class_id = bbox_max_score_classes[idx]
        // bbox = y_bboxes[idx]
        // // clip the coordinate, avoid the value exceed the image boundary.
        // xmin = max(0, int(bbox[0] * width))
        // ymin = max(0, int(bbox[1] * height))
        // xmax = min(int(bbox[2] * width), width)
        // ymax = min(int(bbox[3] * height), height)
        // if draw_result:
        //     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], thickness = tl)
        // if chinese:
        //     image = puttext_chinese(image, id2chiclass[class_id], (xmin, ymin), colors[class_id])  // puttext_chinese
        // else:
        // cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
        //     cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])

        // blob.delete();
        // out.delete();
    } catch (ex) {
        console.log("Error when apply face mask detection");
        console.log(ex);
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

        // Download proto file for caffe model face mask
        await downloadFileAsync(maskMrotoPath, maskProtoDownloadPath);
        console.log("proto downloaded...");

        await downloadFileAsync(maskWeightsPath, maskWeightsDownloadPath);

        console.log("caffemodel downloaded...");

        await sleep(500);

        netDet = cv.readNetFromCaffe('face_detector.prototxt', 'face_detector.caffemodel');

        console.log("net loaded...");

        netMask = cv.readNetFromCaffe(maskMrotoPath, maskWeightsPath);

        console.log("face mask net loaded...");


        netMaskTf = await tf.loadLayersModel('./models/model.json');

        console.log('face mask tf model loaded...');

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
