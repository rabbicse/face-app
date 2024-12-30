import { loginByFace } from "@/api/clients/faceRecognitionClient";
import { FaceRegResponse } from "@/models/responses";

// videoWorker.ts
self.onmessage = async (event) => {
    console.log(`video worker...`);

    // send cropped image python backend
    const response: FaceRegResponse = await loginByFace(event.data.blob);
    console.log(`FRS response: ${JSON.stringify(response)}`);

    // Post the processed frame back to the main thread
    self.postMessage({ status: response != null && response.status == 0 });
};
