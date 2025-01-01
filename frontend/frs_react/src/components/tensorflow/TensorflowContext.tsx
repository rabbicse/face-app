import { createContext, useContext, useEffect, useState } from "react";
import { load, MediaPipeFaceDetectorTfjs } from "@/dnn/face_detector/detector";
import { setupBackend } from "@/dnn/tf-backend";

const TensorFlowContext = createContext<{ model: MediaPipeFaceDetectorTfjs | null }>({ model: null });

export const TensorFlowProvider = ({ children }: { children: React.ReactNode }) => {
    const [model, setModel] = useState<MediaPipeFaceDetectorTfjs | null>(null);

    useEffect(() => {
        const initializeModel = async () => {
            await setupBackend();
            const loadedModel = await load();
            setModel(loadedModel);
        };

        initializeModel();
    }, []);

    return (
        <TensorFlowContext.Provider value={{ model }}>
            {children}
        </TensorFlowContext.Provider>
    );
};

export const useTensorFlowModel = () => useContext(TensorFlowContext);
