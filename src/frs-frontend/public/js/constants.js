// num of landmarks during face detection with blazeface
const NUM_LANDMARKS = 6;
// anchors config during face detection on 
const ANCHORS_CONFIG = {
    'strides': [8, 16],
    'anchors': [2, 6]
};

// blazeface detection config
const config = {
    rotation: 0,
    rotationDegree: 0,
    shiftX: 0,
    shiftY: 0,
    squareLong: true,
    scaleX: 1.5,
    scaleY: 1.5
}

const featureMapSizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]];
const anchorSizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]];
const anchorRatios = [[1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42]];