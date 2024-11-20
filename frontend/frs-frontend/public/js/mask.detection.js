const featureMapSizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]];
const anchorSizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]];
const anchorRatios = [[1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42]];

// decode the output according to anchors
function decodeBBox(anchors, rawOutput, variances = [0.1, 0.1, 0.2, 0.2]) {
    const [anchorXmin, anchorYmin, anchorXmax, anchorYmax] = tf.split(anchors, [1, 1, 1, 1], -1);
    const anchorCX = tf.div(tf.add(anchorXmin, anchorXmax), 2);
    const anchorCY = tf.div(tf.add(anchorYmin, anchorYmax), 2);

    const anchorW = tf.sub(anchorXmax, anchorXmin);
    const anchorH = tf.sub(anchorYmax, anchorYmin);

    const rawOutputScale = tf.mul(rawOutput, tf.tensor(variances));
    const [rawOutputCX, rawOutputCY, rawOutputW, rawOutputH] = tf.split(rawOutputScale, [1, 1, 1, 1], -1);
    const predictCX = tf.add(tf.mul(rawOutputCX, anchorW), anchorCX);
    const predictCY = tf.add(tf.mul(rawOutputCY, anchorH), anchorCY);
    const predictW = tf.mul(tf.exp(rawOutputW), anchorW);
    const predictH = tf.mul(tf.exp(rawOutputH), anchorH);
    const predictXmin = tf.sub(predictCX, tf.div(predictW, 2));
    const predictYmin = tf.sub(predictCY, tf.div(predictH, 2));
    const predictXmax = tf.add(predictCX, tf.div(predictW, 2));
    const predictYmax = tf.add(predictCY, tf.div(predictH, 2));
    // eslint-disable-next-line
    const predictBBox = tf.concat([predictYmin, predictXmin, predictYmax, predictXmax], -1);
    return predictBBox
}

// generate anchors
function anchorGenerator(featureMapSizes, anchorSizes, anchorRatios) {
    let anchorBBoxes = [];
    // eslint-disable-next-line
    featureMapSizes.map((featureSize, idx) => {
        const cx = tf.div(tf.add(tf.linspace(0, featureSize[0] - 1, featureSize[0]), 0.5), featureSize[0]);
        const cy = tf.div(tf.add(tf.linspace(0, featureSize[1] - 1, featureSize[1]), 0.5), featureSize[1]);
        const cxGrid = tf.matMul(tf.ones([featureSize[1], 1]), cx.reshape([1, featureSize[0]]));
        const cyGrid = tf.matMul(cy.reshape([featureSize[1], 1]), tf.ones([1, featureSize[0]]));
        // eslint-disable-next-line
        const cxGridExpend = tf.expandDims(cxGrid, -1);
        // eslint-disable-next-line
        const cyGridExpend = tf.expandDims(cyGrid, -1);
        // eslint-disable-next-line
        const center = tf.concat([cxGridExpend, cyGridExpend], -1);
        const numAnchors = anchorSizes[idx].length + anchorRatios[idx].length - 1;
        const centerTiled = tf.tile(center, [1, 1, 2 * numAnchors]);
        // eslint-disable-next-line
        let anchorWidthHeights = [];

        // eslint-disable-next-line
        for (const scale of anchorSizes[idx]) {
            const ratio = anchorRatios[idx][0];
            const width = scale * Math.sqrt(ratio);
            const height = scale / Math.sqrt(ratio);

            const halfWidth = width / 2;
            const halfHeight = height / 2;
            anchorWidthHeights.push(-halfWidth, -halfHeight, halfWidth, halfHeight);
            // width = tf.mul(scale, tf.sqrt(ratio));
            // height = tf.div(scale, tf.sqrt(ratio));

            // halfWidth = tf.div(width, 2);
            // halfHeight = tf.div(height, 2);
            // anchorWidthHeights.push(tf.neg(halfWidth), tf.neg(halfWidth), halfWidth, halfHeight);
        }

        // eslint-disable-next-line
        for (const ratio of anchorRatios[idx].slice(1)) {
            const scale = anchorSizes[idx][0];
            const width = scale * Math.sqrt(ratio);
            const height = scale / Math.sqrt(ratio);
            const halfWidth = width / 2;
            const halfHeight = height / 2;
            anchorWidthHeights.push(-halfWidth, -halfHeight, halfWidth, halfHeight);
        }
        const bboxCoord = tf.add(centerTiled, tf.tensor(anchorWidthHeights));
        const bboxCoordReshape = bboxCoord.reshape([-1, 4]);
        anchorBBoxes.push(bboxCoordReshape);
    })
    // eslint-disable-next-line
    anchorBBoxes = tf.concat(anchorBBoxes, 0);
    return anchorBBoxes;
}

//  nms function
function nonMaxSuppression(bboxes, confidences, confThresh, iouThresh, width, height, maxOutputSize = 100) {
    const bboxMaxFlag = tf.argMax(confidences, -1);
    const bboxConf = tf.max(confidences, -1);
    const keepIndices = tf.image.nonMaxSuppression(bboxes, bboxConf, maxOutputSize, iouThresh, confThresh);
    // eslint-disable-next-line
    let results = []
    const keepIndicesData = keepIndices.dataSync();
    const bboxConfData = bboxConf.dataSync();
    const bboxMaxFlagData = bboxMaxFlag.dataSync();
    const bboxesData = bboxes.dataSync();
    // eslint-disable-next-line
    keepIndicesData.map((idx) => {
        const xmin = Math.round(Math.max(bboxesData[4 * idx + 1] * width, 0));
        const ymin = Math.round(Math.max(bboxesData[4 * idx + 0] * height, 0));
        const xmax = Math.round(Math.min(bboxesData[4 * idx + 3] * width, width))
        const ymax = Math.round(Math.min(bboxesData[4 * idx + 2] * height, height));
        results.push([[xmin, ymin, xmax, ymax],
        bboxMaxFlagData[idx], bboxConfData[idx]])
    });
    return results;
}


async function detectMask(imgToPredict) {
    const detectionResults = tf.tidy(() => {
        // eslint-disable-next-line
        const width = imgToPredict.width;
        // eslint-disable-next-line
        const height = imgToPredict.height;
        let img = tf.browser.fromPixels(imgToPredict);
        img = tf.image.resizeBilinear(img, [260, 260]);
        img = img.expandDims(0).toFloat().div(tf.scalar(255));
        const [rawBBoxes, rawConfidences] = netMaskTf.predict(img);
        const bboxes = decodeBBox(anchors, tf.squeeze(rawBBoxes));
        const Results = nonMaxSuppression(bboxes, tf.squeeze(rawConfidences), 0.5, 0.5, width, height);
        return Results;
    })
    return detectionResults;
}


let anchors = anchorGenerator(featureMapSizes, anchorSizes, anchorRatios);






//!
async function processFrame(frame, frameBGR) {
    try {

        // detectFaceMask(frameBGR);

        // return;

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

        // cv.imshow(output, frame);

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
        cv.imshow(output, frame);
        frame.delete();
        frameBGR.delete();
    }

    return false;
}


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


//! [Match blob from canvas]
async function matchBlob() {
    try {
        $("#progress").show();

        // get name from input
        let name = $("#name").val();
        // get blob from canvas
        let blob = await getCanvasBlob(faceCanvas);

        // if no blob data found then return
        if (blob === null || blob === undefined) {
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