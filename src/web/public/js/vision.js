const protoDownloadPath = '/models/deploy_lowres.prototxt';
const weightsDownloadPath = '/models/res10_300x300_ssd_iter_140000_fp16.caffemodel';

const maskProtoDownloadPath = '/models/face_mask_detection.prototxt';
const maskWeightsDownloadPath = '/models/face_mask_detection.caffemodel';

const maskProtoTfDownloadPath = '/models/model.json';
const maskWeightsTfDownloadPath = '/models/group1-shard1of1.bin';

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

// IndexedDB
let indexedDB = window.indexedDB || window.webkitIndexedDB || window.mozIndexedDB || window.OIndexedDB || window.msIndexedDB;
let IDBTransaction = window.IDBTransaction || window.webkitIDBTransaction || window.OIDBTransaction || window.msIDBTransaction;
let dbVersion = 1.0;
let dbName = "FRS";
let storeName = "Dnn";



function createDbConnection(dbName, dbVersion) {
    return new Promise(function (resolve) {
        // var request = window.indexedDB.open(dbname)
        const request = indexedDB.open(dbName, dbVersion);
        // create the Contacts object store and indexes
        request.onupgradeneeded = (event) => {
            let db = event.target.result;

            // create the object store 
            // with auto-increment id
            let store = db.createObjectStore('Dnn', {
                autoIncrement: true
            });

            // create an index on the email property
            let index = store.createIndex('model_name', 'model_name', {
                unique: true
            });
        };

        request.onsuccess = (event) => {
            var idb = event.target.result;
            resolve(idb);
        };

        request.onerror = (e) => {
            alert("Enable to access IndexedDB, " + e.target.errorCode)
        };
    });
}

async function insertModel(dbName, dbVersion, storeName, model) {
    let db = await createDbConnection(dbName, dbVersion);
    return new Promise(function (resolve) {
        // create a new transaction
        const txn = db.transaction(storeName, 'readwrite');

        // get the Contacts object store
        const store = txn.objectStore(storeName);
        // put data to store
        let query = store.put(model);

        // handle success case
        query.onsuccess = function (event) {
            console.log(event);
            resolve(event.result);
        };

        // handle the error case
        query.onerror = function (event) {
            console.log(event.target.errorCode);
        }

        // close the database once the 
        // transaction completes
        txn.oncomplete = function () {
            db.close();
        };
    });
}


async function getModelByName(dbName, dbVersion, storeName, name) {
    // create db connection
    let db = await createDbConnection(dbName, dbVersion);

    return new Promise(function (resolve) {
        // create transaction
        const txn = db.transaction(storeName, 'readonly');

        // create object store by store name
        const store = txn.objectStore(storeName);

        // get the index from the Object Store
        const index = store.index('model_name');
        // query by indexes
        let query = index.get(name);

        // return the result object on success
        query.onsuccess = (event) => {
            // console.log(query.result); // result objects
            resolve(query.result);
        };

        query.onerror = (event) => {
            console.log(event.target.errorCode);
        }

        // close the database connection
        txn.oncomplete = function () {
            db.close();
        };
    });
}


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


function getContactById(db, id) {
    const txn = db.transaction('Contacts', 'readonly');
    const store = txn.objectStore('Contacts');

    let query = store.get(id);

    query.onsuccess = (event) => {
        if (!event.target.result) {
            console.log(`The contact with ${id} not found`);
        } else {
            console.table(event.target.result);
        }
    };

    query.onerror = (event) => {
        console.log(event.target.errorCode);
    }

    txn.oncomplete = function () {
        db.close();
    };
};



//! [Initialize DNN]
async function initializeDnn() {
    try {

        // var db = await createDbConnection(dbName, dbVersion);
        // console.log(db);
        let faceDetectionProtoData = await getModelByName(dbName, dbVersion, storeName, protoPath);
        console.log("face detection proto data: ", faceDetectionProtoData);

        // Download proto file for caffe model
        if (faceDetectionProtoData === undefined) {
            await downloadFileAsync(protoPath, protoDownloadPath);
            console.log("proto downloaded...");
        } else {
            saveDnnToFile(faceDetectionProtoData['model_name'], new Uint8Array(faceDetectionProtoData['model']));
        }

        let faceDetectionWeightData = await getModelByName(dbName, dbVersion, storeName, weightsPath);
        console.log("face detection caffe model data: ", faceDetectionWeightData);

        if (faceDetectionWeightData === undefined) {
            await downloadFileAsync(weightsPath, weightsDownloadPath);
            console.log("caffemodel downloaded...");
        } else {
            saveDnnToFile(faceDetectionWeightData['model_name'], new Uint8Array(faceDetectionWeightData['model']));
        }

        let faceMaskDetectionProtoData = await getModelByName(dbName, dbVersion, storeName, 'model.json');
        console.log("face mask detection proto: ", faceMaskDetectionProtoData);
        if (faceMaskDetectionProtoData === undefined) {
            await downloadTfModelAsync('model.json', maskProtoTfDownloadPath);
            console.log("model.json downloaded...");
        }

        let faceMaskDetectionWeightsData = await getModelByName(dbName, dbVersion, storeName, 'group1-shard1of1.bin');
        console.log("face mask detection weights: ", faceMaskDetectionWeightsData);
        if (faceMaskDetectionWeightsData === undefined) {
            await downloadTfModelAsync('group1-shard1of1.bin', maskWeightsTfDownloadPath);
            console.log("tf bin file downloaded...");
        }

        // // Download proto file for caffe model face mask
        // await downloadFileAsync(maskMrotoPath, maskProtoDownloadPath);
        // console.log("proto downloaded...");

        // await downloadFileAsync(maskWeightsPath, maskWeightsDownloadPath);

        // console.log("caffemodel downloaded...");

        await sleep(500);

        netDet = cv.readNetFromCaffe('face_detector.prototxt', 'face_detector.caffemodel');

        console.log("face detection model loaded...");

        // netMask = cv.readNetFromCaffe(maskMrotoPath, maskWeightsPath);

        // console.log("face mask net loaded...");

        // const model = await tf.loadLayersModel('indexeddb://tf-mask-model');
        // console.log("idb model: ", model);

        // netMaskTf = await tf.loadLayersModel('/models/model.json');

        netMaskTf = await tf.loadLayersModel('indexeddb://tf-mask-model');

        // await netMaskTf.save('indexeddb://tf-mask-model');

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

                // insert model to index db
                insertModel(dbName, dbVersion, storeName, { "model_name": path, "model": data });

                saveDnnToFile(path, data);
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


async function downloadTfModelAsync(path, uri) {
    try {
        return $.ajax({
            method: "GET",
            url: uri,
            timeout: 0,
            success: function (response) {
                console.log("download success...");
                let data = new Uint8Array(response);

                // insert model to index db
                insertModel(dbName, dbVersion, storeName, { "model_name": path, "model": data });
            },
            error: function (err) {
                console.log(err);
            }
        });
    } catch (ex) {
        console.log(ex);
    }
}

function saveDnnToFile(path, data) {
    cv.FS_createDataFile('/', path, data, true, false, false);
}

//! [Play webcam using userMedia]
function processWebcamAsync(callback) {
    let cameraFrameWidth = undefined;
    let cameraFrameHeight = undefined;
    // Get a permission from user to use a camera.
    navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false
    }).then(function (stream) {
        stream.getTracks().forEach(function (track) {
            // get track settings
            let settings = track.getSettings();
            // console.log(settings);

            cameraFrameWidth = settings["width"];
            cameraFrameHeight = settings["height"];
            // console.log("w: ", w, " h: ", h);
        });
        
        camera.setAttribute("width", cameraFrameWidth);//camera_output.width);
        camera.setAttribute("height", cameraFrameHeight);//camera_output.height);
        console.log("camera width: ", camera.width, " camera height: ", camera.height);

        camera.srcObject = stream;
        camera.onloadedmetadata = function (e) {
            // play webcam
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
