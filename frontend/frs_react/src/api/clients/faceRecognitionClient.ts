// export const sendCroppedFace = async (faceDataUrl) => {
//     try {
//         // Convert the base64 or data URL to a Blob
//         const blob = await (await fetch(faceDataUrl)).blob();

//         const file = new File([blob], "cropped-face.jpg", { type: "image/jpeg" });

//         // Create a FormData instance
//         const formData = new FormData();
//         // formData.append("photo", blob, "cropped-face.jpg");
//         formData.append("photo", file);        

//         // Send the form data using fetch
//         const response = await fetch("http://localhost:5000/api/v1/dnn/extract-embedding", {
//             method: "POST",
//             body: formData,
//         });

//         // Handle the response
//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }

//         const data = await response.json();
//         console.log("Uploaded face:", data);
//     } catch (error) {
//         console.error("Error uploading face:", error);
//     }
// };

export const sendCroppedFace = async (croppedBlob: Blob) => {
    try {
        const file = new File([croppedBlob], "cropped-face.jpg", { type: "image/jpeg" });
        const formData = new FormData();
        formData.append("photo", file);

        // Send the form data using fetch
        const response = await fetch("http://localhost:5000/api/v1/dnn/extract-embedding", {
            method: "POST",
            body: formData,
        });

        // Handle the response
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Uploaded face:", data);
        return data;
    } catch (error) {
        console.error("Error uploading face:", error);
    }
};
