import 'package:face_app/core/liveness.dart';
import 'package:flutter_test/flutter_test.dart';

List<int> run(BlinkDetector d, List<(double, int)> frames) => [
      for (final (score, t) in frames)
        if (d.update(score, t)) t
    ];

void main() {
  test('detects open -> closed -> open', () {
    final d = BlinkDetector();
    expect(run(d, [(0.1, 0), (0.8, 66), (0.9, 100), (0.1, 133)]), [133]);
    expect(d.hasRecentBlink(1000, 2000), isTrue);
    expect(d.hasRecentBlink(5000, 2000), isFalse);
  });

  test('ignores closed-at-start, long closures, squints and still photos', () {
    expect(run(BlinkDetector(), [(0.9, 0), (0.1, 50)]), isEmpty);
    expect(run(BlinkDetector(), [(0.1, 0), (0.9, 100), (0.9, 1000), (0.1, 1100)]), isEmpty);
    expect(run(BlinkDetector(), [(0.1, 0), (0.45, 100), (0.1, 200)]), isEmpty);
    expect(run(BlinkDetector(), [for (var i = 0; i < 100; i++) (0.12, i * 33)]), isEmpty);
  });

  test('reset forgets blinks', () {
    final d = BlinkDetector();
    run(d, [(0.1, 0), (0.9, 50), (0.1, 100)]);
    d.reset();
    expect(d.hasRecentBlink(200, 5000), isFalse);
  });
}
