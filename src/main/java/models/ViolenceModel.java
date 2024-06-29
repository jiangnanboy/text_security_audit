package models;


import ai.onnxruntime.OrtException;

public class ViolenceModel extends Model {
    public ViolenceModel() {}
    public ViolenceModel(String modelPath) throws OrtException {
        super(modelPath);
    }
}
