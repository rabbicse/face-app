//! [blazeface and face landmark functions]
function createBox(startEndTensor) {
    return {
        startEndTensor,
        startPoint: tf.slice(startEndTensor, [0, 0], [-1, 2]),
        endPoint: tf.slice(startEndTensor, [0, 2], [-1, 2])
    }
};

function scaleBox(box, factors) {
    const starts = tf.mul(box.startPoint, factors);
    const ends = tf.mul(box.endPoint, factors);

    const newCoordinates =
        tf.concat2d([starts, ends], 1);

    return createBox(newCoordinates);
};

function generateAnchors(width, height, outputSpec) {
    const anchors = [];
    for (let i = 0; i < outputSpec.strides.length; i++) {
        const stride = outputSpec.strides[i];
        const gridRows = Math.floor((height + stride - 1) / stride);
        const gridCols = Math.floor((width + stride - 1) / stride);
        const anchorsNum = outputSpec.anchors[i];

        for (let gridY = 0; gridY < gridRows; gridY++) {
            const anchorY = stride * (gridY + 0.5);

            for (let gridX = 0; gridX < gridCols; gridX++) {
                const anchorX = stride * (gridX + 0.5);
                for (let n = 0; n < anchorsNum; n++) {
                    anchors.push([anchorX, anchorY]);
                }
            }
        }
    }

    return anchors;
}


function decodeBounds(boxOutputs, anchors, inputSize) {
    var boxStarts = tf.slice(boxOutputs, [0, 1], [-1, 2]);
    var centers = tf.add(boxStarts, anchors);
    var boxSizes = tf.slice(boxOutputs, [0, 3], [-1, 2]);
    var boxSizesNormalized = tf.div(boxSizes, inputSize);
    var centersNormalized = tf.div(centers, inputSize);
    var halfBoxSize = tf.div(boxSizesNormalized, 2);
    var starts = tf.sub(centersNormalized, halfBoxSize);
    var ends = tf.add(centersNormalized, halfBoxSize);
    var startNormalized = tf.mul(starts, inputSize);
    var endNormalized = tf.mul(ends, inputSize);
    var concatAxis = 1;
    return tf.concat2d([startNormalized, endNormalized], concatAxis);
}



function rectFromBox(box) {
    return {
        xCenter: box.x + box.width / 2,
        yCenter: box.y + box.height / 2,
        width: box.width,
        height: box.height,
    };
}

// https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/rect_transformation_calculator.cc
function transformNormalizedRect(rect, imageSize) {
    let width = rect.width;
    let height = rect.height;
    let rotation = 0;//rect.rotation;

    if (config.rotation != null || config.rotationDegree != null) {
        rotation = computeNewRotation(rotation, config);
    }

    if (rotation === 0) {
        rect.xCenter = rect.xCenter + width * config.shiftX;
        rect.yCenter = rect.yCenter + height * config.shiftY;
    } else {
        const xShift =
            (imageSize.width * width * config.shiftX * Math.cos(rotation) -
                imageSize.height * height * config.shiftY * Math.sin(rotation)) /
            imageSize.width;
        const yShift =
            (imageSize.width * width * config.shiftX * Math.sin(rotation) +
                imageSize.height * height * config.shiftY * Math.cos(rotation)) /
            imageSize.height;
        rect.xCenter = rect.xCenter + xShift;
        rect.yCenter = rect.yCenter + yShift;
    }

    if (config.squareLong) {
        const longSide =
            Math.max(width * imageSize.width, height * imageSize.height);
        width = longSide / imageSize.width;
        height = longSide / imageSize.height;
    } else if (config.squareShort) {
        const shortSide =
            Math.min(width * imageSize.width, height * imageSize.height);
        width = shortSide / imageSize.width;
        height = shortSide / imageSize.height;
    }
    rect.width = width * config.scaleX;
    rect.height = height * config.scaleY;

    return rect;
}

function computeNewRotation(rotation, config) {
    if (config.rotation != null) {
        rotation += config.rotation;
    } else if (config.rotationDegree != null) {
        rotation += Math.PI * config.rotationDegree / 180;
    }
    return normalizeRadians(rotation);
}

function normalizeRadians(angle) {
    return angle - 2 * Math.PI * Math.floor((angle + Math.PI) / (2 * Math.PI));
}

async function tensorsToLandmarks(landmarkTensor, config) {
    flipHorizontally = false;
    flipVertically = false;

    const rawLandmarks = await landmarkTensor.data();
    const numValues = rawLandmarks.length;
    const numDimensions = numValues / 468;

    const outputLandmarks = [];
    for (let ld = 0; ld < 468; ++ld) {
        const offset = ld * numDimensions;
        const landmark = { x: 0, y: 0 };

        if (flipHorizontally) {
            landmark.x = config.inputImageWidth - rawLandmarks[offset];
        } else {
            landmark.x = rawLandmarks[offset];
        }
        if (numDimensions > 1) {
            if (flipVertically) {
                landmark.y = config.inputImageHeight - rawLandmarks[offset + 1];
            } else {
                landmark.y = rawLandmarks[offset + 1];
            }
        }
        if (numDimensions > 2) {
            landmark.z = rawLandmarks[offset + 2];
        }
        if (numDimensions > 3) {
            landmark.score = applyActivation(
                config.visibilityActivation, rawLandmarks[offset + 3]);
        }
        // presence is in rawLandmarks[offset + 4], we don't expose it.

        outputLandmarks.push(landmark);
    }

    for (let i = 0; i < outputLandmarks.length; ++i) {
        const landmark = outputLandmarks[i];
        landmark.x = landmark.x / 192;
        landmark.y = landmark.y / 192;
        // Scale Z coordinate as X + allow additional uniform normalization.
        landmark.z = landmark.z / 192 / 1;
    }
    return outputLandmarks;
}

function calculateLandmarkProjection(landmarks, inputRect) {
    const outputLandmarks = [];
    for (const landmark of landmarks) {
        const x = landmark.x - 0.5;
        const y = landmark.y - 0.5;
        const angle = 0;
        let newX = Math.cos(angle) * x - Math.sin(angle) * y;
        let newY = Math.sin(angle) * x + Math.cos(angle) * y;

        newX = newX * inputRect.width + inputRect.xCenter;
        newY = newY * inputRect.height + inputRect.yCenter;

        const newZ = landmark.z * inputRect.width;  // Scale Z coordinate as x.

        const newLandmark = { ...landmark };

        newLandmark.x = newX;
        newLandmark.y = newY;
        newLandmark.z = newZ;

        outputLandmarks.push(newLandmark);
    }

    return outputLandmarks;
}



//! [face mask detection functions]
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
    return tf.concat([predictYmin, predictXmin, predictYmax, predictXmax], -1);
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
    return tf.concat(anchorBBoxes, 0);
}

//  nms function
async function nonMaxSuppression(bboxes, confidences, confThresh, iouThresh, width, height, maxOutputSize = 100) {
    const bboxMaxFlag = tf.argMax(confidences, -1);
    const bboxConf = tf.max(confidences, -1);
    const keepIndices = await tf.image.nonMaxSuppressionAsync(bboxes, bboxConf, maxOutputSize, iouThresh, confThresh);
    // eslint-disable-next-line
    let results = []
    const keepIndicesData = await keepIndices.data();
    const bboxConfData = await bboxConf.data();
    const bboxMaxFlagData = await bboxMaxFlag.data();
    const bboxesData = await bboxes.data();
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


function parseCvFaceDetectionResults(faces, frame) {
    let rect = faces[0];
    let xTh = rect.width / 3;
    let yTh = rect.height / 3;
    let x = Math.max(rect.x - xTh, 0);
    let y = Math.max(rect.y - yTh, 0);
    let x1 = Math.min(rect.x + rect.width + xTh, frame.cols);
    let y1 = Math.min(rect.y + rect.height + yTh, frame.rows);
    let xCenter = (x1 - x) / 2;
    let yCenter = (y1 - y) / 2;
    let width = x1 - x;
    let height = y1 - y;
    let rct = new cv.Rect(x, y, x1 - x, y1 - y);
    let face = frame.roi(rct);


    return { face: face, rect: { x: x, y: y, xCenter: xCenter, yCenter: yCenter, width: width, height } }
}

function parsetfFaceDetectionResults(faces, frame) {
    var rec = faces[0].bbox;
    let rectBox = rectFromBox(rec);
    let normRect = transformNormalizedRect(rectBox, { width: frame.cols, height: frame.rows });

    let nx1 = Math.max(normRect.xCenter - normRect.width / 2, 0);
    let ny1 = Math.max(normRect.yCenter - normRect.height / 2, 0);
    let nx2 = Math.min(nx1 + normRect.width, frame.cols);
    let ny2 = Math.min(ny1 + normRect.height, frame.rows);
    // cv.rectangle(frame,
    //     { x: Math.floor(nx1), y: Math.floor(ny1) },
    //     { x: Math.floor(nx2), y: Math.floor(ny2) },
    //     [0, 255, 0, 255],
    //     2);
    let nRect = new cv.Rect(Math.floor(nx1),
        Math.floor(ny1),
        Math.ceil(nx2 - nx1),
        Math.ceil(ny2 - ny1));
    let face = frame.roi(nRect);

    // extract face landmark
    normRect.xCenter = (nx2 - nx1) / 2;
    normRect.yCenter = (ny2 - ny1) / 2;
    normRect.width = nx2 - nx1;
    normRect.height = ny2 - ny1;

    return { face: face, rect: normRect };
}