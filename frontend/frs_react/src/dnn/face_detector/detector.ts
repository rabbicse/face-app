/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import { FaceDetector } from './face_detector';
import { convertImageToTensor } from '../shared/convert_image_to_tensor';
import { createSsdAnchors } from '../shared/create_ssd_anchors';
import { detectionProjection } from '../shared/detection_projection';
import { detectorResult } from '../shared/detector_result';
import { toImageTensor } from '../shared/image_utils';
import { ImageToTensorConfig, TensorsToDetectionsConfig } from '../shared/interfaces/config';
import { Rect } from '../shared/interfaces/shapes';
import { AnchorTensor, Detection } from '../shared/interfaces/shapes';
import { nonMaxSuppression } from '../shared/non_max_suppression';
import { tensorsToDetections } from '../shared/tensors_to_detections';
import * as constants from './constants'
import { Face, FaceDetectorInput, MediaPipeFaceDetectorTfjsEstimationConfig, MediaPipeFaceDetectorTfjsModelConfig } from '../shared/types'
import { MEDIAPIPE_FACE_DETECTOR_KEYPOINTS } from './constants';
import { ImageSize } from '../shared/interfaces/common';

function getInputTensorDimensions(input: tf.Tensor3D | ImageData | HTMLVideoElement |
  HTMLImageElement |
  HTMLCanvasElement): [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
    [input.height, input.width];
}

export class MediaPipeFaceDetectorTfjs implements FaceDetector {
  private readonly imageToTensorConfig: ImageToTensorConfig;
  private readonly tensorsToDetectionConfig: TensorsToDetectionsConfig;
  private readonly anchors: Rect[];
  private readonly anchorTensor: AnchorTensor;

  constructor(
    detectorModelType: 'short' | 'full',
    private readonly detectorModel: tfconv.GraphModel,
    private readonly maxFaces: number) {
    if (detectorModelType === 'full') {
      this.imageToTensorConfig = constants.FULL_RANGE_IMAGE_TO_TENSOR_CONFIG;
      this.tensorsToDetectionConfig =
        constants.FULL_RANGE_TENSORS_TO_DETECTION_CONFIG;
      this.anchors =
        createSsdAnchors(constants.FULL_RANGE_DETECTOR_ANCHOR_CONFIG);
    } else {
      this.imageToTensorConfig = constants.SHORT_RANGE_IMAGE_TO_TENSOR_CONFIG;
      this.tensorsToDetectionConfig =
        constants.SHORT_RANGE_TENSORS_TO_DETECTION_CONFIG;
      this.anchors =
        createSsdAnchors(constants.SHORT_RANGE_DETECTOR_ANCHOR_CONFIG);
    }

    const anchorW = tf.tensor1d(this.anchors.map(a => a.width));
    const anchorH = tf.tensor1d(this.anchors.map(a => a.height));
    const anchorX = tf.tensor1d(this.anchors.map(a => a.xCenter));
    const anchorY = tf.tensor1d(this.anchors.map(a => a.yCenter));
    this.anchorTensor = { x: anchorX, y: anchorY, w: anchorW, h: anchorH };
  }

  dispose() {
    this.detectorModel.dispose();
    tf.dispose([
      this.anchorTensor.x, this.anchorTensor.y, this.anchorTensor.w,
      this.anchorTensor.h
    ]);
  }

  reset() { }

  // Detects faces.
  // Subgraph: FaceDetectionShort/FullRangeCpu.
  // ref:
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_short_range_cpu.pbtxt
  // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_detection/face_detection_full_range_cpu.pbtxt
  async detectFaces(image: FaceDetectorInput, flipHorizontal = false):
    Promise<Detection[]> {
    if (image == null) {
      this.reset();
      return [];
    }

    const image3d = tf.tidy(() => {
      let imageTensor = tf.cast(toImageTensor(image), 'float32');
      if (flipHorizontal) {
        const batchAxis = 0;
        imageTensor = tf.squeeze(
          tf.image.flipLeftRight(
            // tslint:disable-next-line: no-unnecessary-type-assertion
            tf.expandDims(imageTensor, batchAxis) as tf.Tensor4D),
          [batchAxis]);
      }
      return imageTensor;
    });

    // FaceDetectionShort/FullRangeModelCpu: ImageToTensorCalculator
    // Transforms the input image into a 128x128 tensor while keeping the aspect
    // ratio (what is expected by the corresponding face detection model),
    // resulting in potential letterboxing in the transformed image.
    const { imageTensor: inputTensors, transformationMatrix: transformMatrix } =
      convertImageToTensor(image3d, this.imageToTensorConfig);

    const detectionResult =
      this.detectorModel.execute(inputTensors, 'Identity:0') as tf.Tensor3D;
    // FaceDetectionShort/FullRangeModelCpu: InferenceCalculator
    // The model returns a tensor with the following shape:
    // [1 (batch), 896 (anchor points), 17 (data for each anchor)]
    const { boxes, logits } = detectorResult(detectionResult);
    // FaceDetectionShort/FullRangeModelCpu: TensorsToDetectionsCalculator
    const unfilteredDetections: Detection[] = await tensorsToDetections(
      [logits, boxes], this.anchorTensor, this.tensorsToDetectionConfig);

    if (unfilteredDetections.length === 0) {
      tf.dispose([image3d, inputTensors, detectionResult, logits, boxes]);
      return unfilteredDetections;
    }

    // FaceDetectionShort/FullRangeModelCpu: NonMaxSuppressionCalculator
    const filteredDetections = await nonMaxSuppression(
      unfilteredDetections, this.maxFaces,
      constants.DETECTOR_NON_MAX_SUPPRESSION_CONFIG.minSuppressionThreshold,
      constants.DETECTOR_NON_MAX_SUPPRESSION_CONFIG.overlapType);

    const detections =
      // FaceDetectionShortRangeModelCpu:
      // DetectionProjectionCalculator
      detectionProjection(filteredDetections, transformMatrix);

    tf.dispose([image3d, inputTensors, detectionResult, logits, boxes]);

    return detections;
  }

  async estimateFaces(
    image: tf.Tensor3D | ImageData | HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
    imageSize: ImageSize,
    estimationConfig?: MediaPipeFaceDetectorTfjsEstimationConfig):
    Promise<Face[]> {
    // console.log(`width: ${imageSize.width} height: ${imageSize.height}`);
    // const imageSize = getInputTensorDimensions(image);
    // console.log(imageSize);
    const flipHorizontal =
      estimationConfig ? estimationConfig.flipHorizontal : false;
    return this.detectFaces(image, flipHorizontal)
      .then(detections => detections.map(detection => {
        const keypoints = detection.locationData.relativeKeypoints.map(
          (keypoint, i) => ({
            ...keypoint,
            x: keypoint.x * imageSize.width,
            y: keypoint.y * imageSize.height,
            name: MEDIAPIPE_FACE_DETECTOR_KEYPOINTS[i]
          }));
        const box = detection.locationData.relativeBoundingBox;
        if (box !== undefined) {
          for (const key of ['width', 'xMax', 'xMin'] as const) {
            box[key] *= imageSize.width;
          }
          for (const key of ['height', 'yMax', 'yMin'] as const) {
            box[key] *= imageSize.height;
          }
        }
        return { keypoints, box };
      }));
  }
}

/**
 * Loads the MediaPipeFaceDetector model.
 *
 */
export async function load() {
  const detectorModel = await tfconv.loadGraphModel(
    constants.DEFAULT_DETECTOR_MODEL_URL_FULL_SPARSE, { fromTFHub: true });

  return new MediaPipeFaceDetectorTfjs('full', detectorModel, 100);
}