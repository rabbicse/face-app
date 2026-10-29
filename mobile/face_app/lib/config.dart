/// Build time configuration: flutter run --dart-define=FRS_API=http://192.168.1.10:5000
class AppConfig {
  /// Backend base URL. The Android emulator reaches the host machine through 10.0.2.2.
  static const apiBaseUrl = String.fromEnvironment('FRS_API', defaultValue: 'http://10.0.2.2:5000');

  /// Must match the web app and the backend (FRS_EMBEDDING_MODELS).
  static const embeddingModelId = 'w600k_mbf';
  static const embeddingModelAsset = 'assets/models/w600k_mbf.onnx';
  static const embeddingInputSize = 112;

  /// Average the embedding of the face and of its mirror image (+accuracy, 2x compute).
  static const flipTta = true;

  /// Upper bound of embeddings computed per second while capturing.
  static const maxEmbeddingsPerSecond = 6;
}
