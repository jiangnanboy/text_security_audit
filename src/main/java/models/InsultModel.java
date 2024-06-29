package models;

import ai.onnxruntime.OrtException;

public class InsultModel extends Model {
    public InsultModel() {}
    public InsultModel(String modelPath) throws OrtException {
        super(modelPath);
    }
}
