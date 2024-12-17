import { Person } from "@/models/person";

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

export const registerFace = async (blob: Blob, person: Person) => {
    try {
        const photo = new File([blob], "face.jpg", { type: "image/jpeg" });
        const formData = new FormData();
        formData.append("photo", photo);

        formData.append("person_id", person.personId);//`${person.personId}`);
        formData.append("name", `${person.name}`);
        formData.append("email", `${person.email}`);
        formData.append("phone", `${person.phone}`);
        formData.append("age", person.age);
        formData.append("city", `${person.city}`);
        formData.append("country", `${person.country}`);
        formData.append("address", `Test Address`);

        console.log(`Form data: ${JSON.stringify(formData)}`);


        // Send the form data using fetch
        const response = await fetch("http://localhost:5000/api/v1/register/", {
            method: "POST",
            headers: {
                Accept: "application/json",
                // Do not set 'Content-Type', fetch will handle it with FormData
            },
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
