package models;

import ai.onnxruntime.OrtException;

public class PornModel extends Model {
    public PornModel() {}
    public PornModel(String modelPath) throws OrtException {
        super(modelPath);
    }
}
