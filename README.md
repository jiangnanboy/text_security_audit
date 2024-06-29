### text security audit 安全审核-语义模型过滤 敏感内容检测系统

本项目收集大量的涉及政治、色情、辱骂、暴力违禁等敏感数据，据此微调roberta模型，根据模型对文本内容进行分类打分，进行检测审核；

针对业务场景下个性化的数据和需求，可自由定制审核模型的阈值参数，合适的策略配置将有效提升内容审核的召回率和精确率。

This project collects a large number of sensitive data involving politics, pornography, abuse, violence and contraband, and fine-tunes roberta model according to which the text content is classified and scored for detection and verification.

You can customize the threshold parameters of the audit model according to the personalized data and requirements in business scenarios. Proper policy configuration can effectively improve the recall rate and accuracy rate of content audit.

-----------------------------------------------------------------------
将roberta类模型转为onnx格式，利用java进行推理。

The roberta model is converted to onnx format, and java is used for inference.

* 模型转为onnx见https://github.com/jiangnanboy/model2onnx

模型下载：

链接: https://pan.baidu.com/s/1bksb12LOUV3dhJd0Wk4ZAw 提取码: 2dkf

将下载后的model放在resources下。Place the downloaded model under resources.
### usage
【text_classification/TextClassify】

``` java
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
```

### requirement
java11+

onnxruntime1.11.0

### contact
- github：https://github.com/jiangnanboy

### reference
- https://github.com/jiangnanboy/model2onnx
- https://github.com/jiangnanboy/java_textcnn_onnx
- https://github.com/jiangnanboy/ad_detection
- https://huggingface.co/thu-coai/roberta-base-cold

