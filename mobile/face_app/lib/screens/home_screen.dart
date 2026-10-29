import 'package:flutter/material.dart';

import '../api/api_client.dart';
import '../config.dart';
import '../core/registration.dart';
import 'capture_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _name = TextEditingController();
  final _email = TextEditingController();

  @override
  void dispose() {
    _name.dispose();
    _email.dispose();
    super.dispose();
  }

  void _register(List<PoseRequirement> plan) {
    final name = _name.text.trim();
    final email = _email.text.trim();
    if (name.isEmpty || email.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Enter your name and e-mail first')));
      return;
    }
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => CaptureScreen.register(
        person: PersonInfo(personId: personIdFromEmail(email), name: name, email: email),
        plan: plan,
      ),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Recognition')),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          const Text(
            'Faces are detected, aligned and converted to a 512-d embedding on this device. '
            'Only the embedding is sent to the server.',
          ),
          const SizedBox(height: 24),
          TextField(controller: _name, decoration: const InputDecoration(labelText: 'Full name')),
          TextField(
            controller: _email,
            keyboardType: TextInputType.emailAddress,
            decoration: const InputDecoration(labelText: 'E-mail'),
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            icon: const Icon(Icons.face_retouching_natural),
            label: const Text('Register face (guided, 5 poses)'),
            onPressed: () => _register(fullRegistrationPlan),
          ),
          TextButton(
            onPressed: () => _register(quickRegistrationPlan),
            child: const Text('Quick registration (frontal only)'),
          ),
          const Divider(height: 48),
          FilledButton.tonalIcon(
            icon: const Icon(Icons.face),
            label: const Text('Identify me'),
            onPressed: () =>
                Navigator.of(context).push(MaterialPageRoute(builder: (_) => const CaptureScreen.identify())),
          ),
          const SizedBox(height: 24),
          Text('API: ${AppConfig.apiBaseUrl}', style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}
