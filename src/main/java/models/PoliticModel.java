package models;

import ai.onnxruntime.OrtException;

public class PoliticModel extends Model {
    public PoliticModel(){}
    public PoliticModel(String modelPath) throws OrtException {
        super(modelPath);
    }
}
