import 'dart:io' show Platform;

import 'package:camera/camera.dart';
import 'package:flutter/services.dart' show DeviceOrientation;

import 'camera_frame.dart';

const _orientationDegrees = {
  DeviceOrientation.portraitUp: 0,
  DeviceOrientation.landscapeLeft: 90,
  DeviceOrientation.portraitDown: 180,
  DeviceOrientation.landscapeRight: 270,
};

/// Clockwise rotation that makes the frame upright (same logic as the google_mlkit examples).
int frameRotation(CameraDescription camera, DeviceOrientation deviceOrientation) {
  if (Platform.isIOS) return camera.sensorOrientation;
  final device = _orientationDegrees[deviceOrientation] ?? 0;
  return camera.lensDirection == CameraLensDirection.front
      ? (camera.sensorOrientation + device) % 360
      : (camera.sensorOrientation - device + 360) % 360;
}

/// Wraps a camera plugin image without copying the pixel data.
CameraFrame? cameraFrameFrom(CameraImage image, int rotation) {
  final PixelLayout layout;
  switch (image.format.group) {
    case ImageFormatGroup.nv21:
      layout = PixelLayout.nv21;
    case ImageFormatGroup.bgra8888:
      layout = PixelLayout.bgra8888;
    case ImageFormatGroup.yuv420:
      // camera_android_camerax reports NV21 data as yuv420 with a single plane
      layout = image.planes.length == 1 ? PixelLayout.nv21 : PixelLayout.yuv420;
    default:
      return null;
  }
  return CameraFrame(
    width: image.width,
    height: image.height,
    layout: layout,
    rotation: rotation,
    planes: [for (final p in image.planes) FramePlane(p.bytes, p.bytesPerRow, p.bytesPerPixel ?? 1)],
  );
}

/// NV21 on Android (what ML Kit wants), BGRA on iOS.
ImageFormatGroup get preferredImageFormat => Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888;
