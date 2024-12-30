// videoWorker.ts
self.onmessage = async (event) => {
    console.log(`video worker...`);
    console.log(event);
    // const { frameData, width, height } = event.data;    
    const status = true;

    // Post the processed frame back to the main thread
    self.postMessage({ status: status });
};
