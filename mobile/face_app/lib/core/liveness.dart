/// Detects an open -> closed -> open eye cycle (port of frontend/.../core/liveness.ts). A printed photo or
/// a still image cannot blink, so identification requires one recent blink. Not a defence against replayed
/// videos or a tampered app.
class BlinkDetector {
  BlinkDetector({this.openBelow = 0.3, this.closedAbove = 0.5, this.maxClosedMs = 700});

  /// eyes count as open below this "closed" score (0 = wide open, 1 = shut)
  final double openBelow;
  final double closedAbove;

  /// a longer closure is not a blink
  final int maxClosedMs;

  String _state = 'unknown';
  int _closedSince = 0;
  int? _lastBlink;
  int _count = 0;

  int get count => _count;

  /// [closedScore]: how closed the *less* closed eye is (0..1). Returns true when a blink completes.
  bool update(double closedScore, int timestampMs) {
    if (closedScore < openBelow) {
      final blinked = _state == 'closed' && timestampMs - _closedSince <= maxClosedMs;
      _state = 'open';
      if (blinked) {
        _lastBlink = timestampMs;
        _count++;
      }
      return blinked;
    }
    if (closedScore > closedAbove && _state == 'open') {
      _state = 'closed';
      _closedSince = timestampMs;
    }
    return false;
  }

  bool hasRecentBlink(int timestampMs, int windowMs) => _lastBlink != null && timestampMs - _lastBlink! <= windowMs;

  /// Call when the face is lost, so a blink of one face cannot unlock another.
  void reset() {
    _state = 'unknown';
    _lastBlink = null;
  }
}
