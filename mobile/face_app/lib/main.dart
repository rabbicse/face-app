import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/home_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // the pose math assumes the upright frame; keep the UI in portrait
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const FaceApp());
}

class FaceApp extends StatelessWidget {
  const FaceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Recognition',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}
