package text_classification;
import models.*;


public class TextClassify {

    public static void main(String[] args) throws Exception {
        var textTokenizer = new TextTokenizer("resources\\vocab.txt");
        var politicModel = new PoliticModel("resources\\roberta_wwm_politic_model.onnx");
        var violenceModel = new ViolenceModel("resources\\roberta_wwm_violence_model.onnx");
        var pornModel = new PornModel("resources\\roberta_wwm_porn_model.onnx");
        var insultModel = new InsultModel("resources\\roberta_wwm_insult_model.onnx");


        var text = "黑人很多都好吃懒做，偷奸耍滑！";

        var onnxTensorMap = textTokenizer.parseInputText(text, politicModel.env);

        // politic detection
        var pairResult = politicModel.pred(onnxTensorMap);
        System.out.println(pairResult);

        // violence detection
        pairResult = violenceModel.pred(onnxTensorMap);
        System.out.println(pairResult);

        // porn detection
        pairResult = pornModel.pred(onnxTensorMap);
        System.out.println(pairResult);

        // insult detection
        pairResult = insultModel.pred(onnxTensorMap);
        System.out.println(pairResult);

        /**
         * label=0 -> No; label=1 -> Yes
         *
         * politic:
         * (1,0.77812237)
         *
         * violence:
         * (1,0.7366322)
         *
         * porn:
         * (0,0.65102273)
         *
         * insult:
         * (0,0.6051175)
         */
    }


}

