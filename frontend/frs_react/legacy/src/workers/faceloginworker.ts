import { loginByFace } from "@/api/clients/faceRecognitionClient";
import { FaceRegResponse } from "@/models/responses";

let queue: any[] = [];

// videoWorker.ts
self.onmessage = async (event) => {
    // Add new data to the queue
    queue.push(event.data.blob);

    // Keep only the last two items
    if (queue.length > 1) {
        queue.shift();
    }


    if (queue.length > 0) {
        console.log(`queue size: ${queue.length}`);
        // send cropped image python backend
        const response: FaceRegResponse = await loginByFace(queue.pop());
        console.log(`FRS response: ${JSON.stringify(response)}`);

        console.log(`Person: ${JSON.stringify(response.result.payload)}`);

        // Post the processed frame back to the main thread
        self.postMessage({ status: response != null && response.status == 0, result: JSON.stringify(response.result.payload) });
    }
};
