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
import {io} from '@tensorflow/tfjs-core';
import { Keypoint, PixelInput } from '../shared/interfaces/common';
import { BoundingBox } from '../shared/interfaces/shapes';

export type { Keypoint };

export enum SupportedModels {
    MediaPipeFaceMesh = 'MediaPipeFaceMesh',
}

/**
 * Common config to create the face detector.
 *
 * `maxFaces`: Optional. Default to 1. The maximum number of faces that will
 * be detected by the model. The number of returned faces can be less than the
 * maximum (for example when no faces are present in the input).
 */
export interface ModelConfig {
    maxFaces?: number;
}

/**
 * Common config for the `estimateFaces` method.
 *
 * `flipHorizontal`: Optional. Default to false. In some cases, the image is
 * mirrored, e.g. video stream from camera, flipHorizontal will flip the
 * keypoints horizontally.
 *
 * `staticImageMode`: Optional. Default to true. If set to true, face detection
 * will run on every input image, otherwise if set to false then detection runs
 * once and then the model simply tracks those landmarks without invoking
 * another detection until it loses track of any of the faces (ideal for
 * videos).
 */
export interface EstimationConfig {
    flipHorizontal?: boolean;
    staticImageMode?: boolean;
}

/**
 * Allowed input format for the `estimateFaces` method.
 */
export type FaceLandmarksDetectorInput = PixelInput;

export interface Face {
    keypoints: Keypoint[];  // Points of mesh in the detected face.
    // MediaPipeFaceMesh has 468 keypoints.
    box: BoundingBox;       // A bounding box around the detected face.
}

/**
 * Common MediaPipeFaceMesh model config.
 */
export interface MediaPipeFaceMeshModelConfig extends ModelConfig {
    runtime: 'mediapipe'|'tfjs';
    refineLandmarks: boolean;
  }
  
  export interface MediaPipeFaceMeshEstimationConfig extends EstimationConfig {}
  
  /**
   * Model parameters for MediaPipeFaceMesh MediaPipe runtime
   *
   * `runtime`: Must set to be 'mediapipe'.
   *
   * `refineLandmarks`: If set to true, refines the landmark coordinates around
   * the eyes and lips, and output additional landmarks around the irises.
   *
   * `solutionPath`: Optional. The path to where the wasm binary and model files
   * are located.
   */
  export interface MediaPipeFaceMeshMediaPipeModelConfig extends
      MediaPipeFaceMeshModelConfig {
    runtime: 'mediapipe';
    solutionPath?: string;
  }
  
  /**
   * Face estimation parameters for MediaPipeFaceMesh MediaPipe runtime.
   */
  export interface MediaPipeFaceMeshMediaPipeEstimationConfig extends
      MediaPipeFaceMeshEstimationConfig {}

/**
 * Model parameters for MediaPipeFaceMesh TFJS runtime.
 *
 * `runtime`: Must set to be 'tfjs'.
 *
 * `refineLandmarks`: Defaults to false. If set to true, refines the landmark
 * coordinates around the eyes and lips, and output additional landmarks around
 * the irises.
 *
 * `detectorModelUrl`: Optional. An optional string that specifies custom url of
 * the detector model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 *
 * `landmarkModelUrl`: Optional. An optional string that specifies custom url of
 * the landmark model. This is useful for area/countries that don't have access
 * to the model hosted on tf.hub.
 */
export interface MediaPipeFaceMeshTfjsModelConfig extends
    MediaPipeFaceMeshModelConfig {
  runtime: 'tfjs';
  detectorModelUrl?: string|io.IOHandler;
  landmarkModelUrl?: string|io.IOHandler;
}

/**
 * Face estimation parameters for MediaPipeFaceMesh TFJS runtime.
 */
export interface MediaPipeFaceMeshTfjsEstimationConfig extends
    MediaPipeFaceMeshEstimationConfig {}