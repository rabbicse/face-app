import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';
import * as tf from "@tensorflow/tfjs";

export async function setupBackend() {
    const isWebGPUSupported = await tf.setBackend('webgpu');
    if (!isWebGPUSupported) {
        console.log('WebGPU is not supported. Falling back to WebGL.');
        await tf.setBackend('webgl'); // Fallback to webgl
    }
    await tf.ready();
    console.log(`Using backend: ${tf.getBackend()}`);
}