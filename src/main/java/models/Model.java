package models;

import ai.onnxruntime.*;
import org.apache.commons.lang3.tuple.Pair;
import util.CollectionUtil;

import java.util.List;
import java.util.Map;
import java.util.Optional;

public class Model {
    public OrtSession session;
    public OrtEnvironment env;

    public Model() {}

    public Model(String modelPath) throws OrtException {
        loadModel(modelPath);
    }

    /**
     * load model
     * @throws OrtException
     */
    public void loadModel(String modelPath) throws OrtException {
        System.out.println("load model...");
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    /**
     * close model
     */
    public void closeModel() {
        System.out.println("close model...");
        if (Optional.of(session).isPresent()) {
            try {
                session.close();
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }
        if(Optional.of(env).isPresent()) {
            env.close();
        }
    }


    public Pair<Integer, Float> pred(Map<String, OnnxTensor> onnxTensorMap) {
        Pair<Integer, Float> pairResult = null;
        try{
            try(var results = session.run(onnxTensorMap)) {
                var onnxValue = results.get(0);
                var labels = (float[][]) onnxValue.getValue();
                var maskLables = labels[0];
                maskLables = softmax(maskLables);
                pairResult = predMax(maskLables);
//                System.out.println(pairResult.getLeft());
//                System.out.println(pairResult.getRight());

            }
        } catch (OrtException e) {
            e.printStackTrace();
        }
        return pairResult;
    }

    public Pair<Integer, Float> predMax(float[] probabilities) {
        var maxVal = Float.NEGATIVE_INFINITY;
        var idx = 0;
        for (var i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i];
                idx = i;
            }
        }
        return Pair.of(idx, maxVal);
    }

    public float[] softmax(float[] input) {
        List<Float> inputList = CollectionUtil.newArrayList();
        for(var i=0; i<input.length; i++) {
            inputList.add(input[i]);
        }
        var inputSum = inputList.stream().mapToDouble(Math::exp).sum();
        var output = new float[input.length];
        for (var i=0; i<input.length; i++) {
            output[i] = (float) (Math.exp(input[i]) / inputSum);
        }
        return output;
    }

}
