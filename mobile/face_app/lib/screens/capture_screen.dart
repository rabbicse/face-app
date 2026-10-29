import 'dart:async';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../config.dart';
import '../core/identification.dart';
import '../core/pose.dart';
import '../core/quality.dart';
import '../core/registration.dart';
import '../vision/camera_adapter.dart';
import '../vision/face_engine.dart';
import '../vision/face_observation.dart';
import '../widgets/face_overlay.dart';

enum CaptureMode { register, identify }

const _poseNames = {
  PoseLabel.frontal: 'Front',
  PoseLabel.turnLeft: 'Side',
  PoseLabel.turnRight: 'Side',
  PoseLabel.lookUp: 'Up',
  PoseLabel.lookDown: 'Down',
};

/// Camera screen shared by registration (guided poses) and identification (no pose requirement).
class CaptureScreen extends StatefulWidget {
  const CaptureScreen.register({super.key, required PersonInfo this.person, required this.plan})
      : mode = CaptureMode.register;

  const CaptureScreen.identify({super.key})
      : mode = CaptureMode.identify,
        person = null,
        plan = const [];

  final CaptureMode mode;
  final PersonInfo? person;
  final List<PoseRequirement> plan;

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  CameraController? _controller;
  FaceEngine? _engine;
  final _api = FaceApiClient();
  late final RegistrationSession _session = RegistrationSession(plan: widget.plan);
  final _aggregator = IdentificationAggregator();
  final _clock = Stopwatch()..start();

  bool _busy = false;
  bool _finished = false;
  bool _queryInFlight = false;
  int _lastEmbedMs = -100000;
  int _nextQueryMs = 0;

  String _message = 'Starting camera...';
  FaceObservation? _face;
  Color _tone = Colors.white;
  Size _frameSize = Size.zero;
  String? _error;
  PersonInfo? _matched;

  bool get _registering => widget.mode == CaptureMode.register;

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    try {
      final cameras = await availableCameras();
      final camera = cameras.firstWhere((c) => c.lensDirection == CameraLensDirection.front,
          orElse: () => cameras.first);
      final controller = CameraController(camera, ResolutionPreset.medium,
          enableAudio: false, imageFormatGroup: preferredImageFormat);
      await controller.initialize();
      final engine = await FaceEngine.create(accuratePose: _registering);
      if (!mounted) {
        await controller.dispose();
        await engine.close();
        return;
      }
      setState(() {
        _controller = controller;
        _engine = engine;
        _message = _registering ? _session.currentTarget!.instruction : 'Look at the camera';
      });
      await controller.startImageStream((image) {
        // drop frames while the previous one is processed: latency stays at one frame
        if (_busy || _finished) return;
        _busy = true;
        _process(image, camera)
            .catchError((Object e) => debugPrint('frame processing failed: $e'))
            .whenComplete(() => _busy = false);
      });
    } catch (e) {
      setState(() => _error = 'Camera / model initialisation failed: $e');
    }
  }

  Future<void> _process(CameraImage image, CameraDescription camera) async {
    final engine = _engine, controller = _controller;
    if (engine == null || controller == null) return;
    final frame = cameraFrameFrom(image, frameRotation(camera, controller.value.deviceOrientation));
    if (frame == null) return;

    final analysis = await engine.analyze(frame);
    final now = _clock.elapsedMilliseconds;
    var message = _message;
    var tone = Colors.white;

    if (analysis == null) {
      message = 'No face detected';
      _aggregator.reset();
    } else if (_registering) {
      final target = _session.currentTarget;
      if (target == null) return;
      if (!_session.wants(analysis.face.pose, now)) {
        message = target.instruction;
        tone = Colors.amber;
      } else if (!analysis.quality.ok) {
        message = qualityMessages[analysis.quality.issues.first]!;
        tone = Colors.amber;
      } else {
        tone = Colors.greenAccent;
        message = '${target.instruction} - hold still';
        if (now - _lastEmbedMs >= 1000 ~/ AppConfig.maxEmbeddingsPerSecond) {
          _lastEmbedMs = now;
          final embedding = await engine.embed(analysis);
          final result = _session.offer(
              embedding: embedding, pose: analysis.face.pose, quality: analysis.quality.score, timestampMs: now);
          if (result == OfferResult.inconsistent) message = 'Only one person please';
          if (result == OfferResult.complete) unawaited(_submitRegistration());
        }
      }
    } else {
      // identification: quality gate only, no head pose requirement
      if (!analysis.quality.ok) {
        message = qualityMessages[analysis.quality.issues.first]!;
        tone = Colors.amber;
      } else {
        tone = Colors.greenAccent;
        if (!_queryInFlight && now >= _nextQueryMs && now - _lastEmbedMs >= 1000 ~/ AppConfig.maxEmbeddingsPerSecond) {
          _lastEmbedMs = now;
          final fused = _aggregator.add(await engine.embed(analysis), now);
          if (fused != null) unawaited(_identify(fused));
        }
      }
    }

    if (!mounted || _finished) return;
    setState(() {
      _face = analysis?.face;
      _frameSize = Size(frame.uprightWidth.toDouble(), frame.uprightHeight.toDouble());
      _message = message;
      _tone = tone;
    });
  }

  Future<void> _identify(Float32List fused) async {
    _queryInFlight = true;
    try {
      final res = await _api.identify(fused);
      if (res.matched && res.person != null) {
        _finish();
        setState(() {
          _matched = res.person;
          _message = 'Welcome, ${res.person!.name}! (score ${res.score?.toStringAsFixed(2)})';
        });
      } else {
        _nextQueryMs = _clock.elapsedMilliseconds + 700;
        if (mounted) setState(() => _message = res.reason == 'no_enrolled_faces' ? 'No faces registered yet' : 'Face not recognized');
      }
    } catch (e) {
      _nextQueryMs = _clock.elapsedMilliseconds + 2000;
      if (mounted) setState(() => _message = 'Server error: $e');
    } finally {
      _queryInFlight = false;
    }
  }

  Future<void> _submitRegistration() async {
    _finish();
    setState(() => _message = 'Saving your face profile...');
    try {
      await _api.register(widget.person!, _session.collected);
      if (mounted) setState(() => _message = 'Face enrolled successfully!');
    } catch (e) {
      if (mounted) setState(() => _error = 'Registration failed: $e');
    }
  }

  void _finish() {
    _finished = true;
    final c = _controller;
    if (c != null && c.value.isStreamingImages) unawaited(c.stopImageStream());
  }

  @override
  void dispose() {
    _finished = true;
    _controller?.dispose();
    _engine?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      appBar: AppBar(title: Text(_registering ? 'Register ${widget.person!.name}' : 'Identify')),
      body: _error != null
          ? Center(child: Padding(padding: const EdgeInsets.all(24), child: Text(_error!)))
          : Column(
              children: [
                Expanded(
                  child: Center(
                    child: controller == null || !controller.value.isInitialized
                        ? const CircularProgressIndicator()
                        : AspectRatio(
                            aspectRatio: 1 / controller.value.aspectRatio,
                            child: Stack(fit: StackFit.expand, children: [
                              CameraPreview(controller),
                              CustomPaint(
                                painter: FaceOverlayPainter(
                                  face: _face,
                                  frameSize: _frameSize,
                                  mirror: controller.description.lensDirection == CameraLensDirection.front,
                                  color: _tone,
                                  progress: _registering ? _session.fraction : null,
                                ),
                              ),
                              if (_matched != null || (_registering && _session.isComplete))
                                const ColoredBox(
                                    color: Colors.black45,
                                    child: Center(child: Icon(Icons.check_circle, color: Colors.greenAccent, size: 96))),
                            ]),
                          ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text(_message, textAlign: TextAlign.center, style: Theme.of(context).textTheme.titleMedium),
                ),
                if (_registering)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 24),
                    child: Wrap(spacing: 8, children: [
                      for (final r in widget.plan)
                        Chip(
                          label: Text('${_poseNames[r.label]} ${_session.count(r.label)}/${r.count}'),
                          backgroundColor: _session.count(r.label) >= r.count ? Colors.green.shade200 : null,
                        ),
                    ]),
                  ),
              ],
            ),
    );
  }
}
