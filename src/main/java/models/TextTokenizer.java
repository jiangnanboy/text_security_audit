package models;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtUtil;
import bert_tokenizer.tokenizerimpl.BertTokenizer;

import java.util.List;
import java.util.Map;

public class TextTokenizer {
    BertTokenizer tokenizer;

    public TextTokenizer() {}
    public TextTokenizer(String vocabPath) {
        loadVocab(vocabPath);
    }
    public void loadVocab(String vocabPath) {
        tokenizer = new BertTokenizer(vocabPath);
    }

    /**
     * tokenize text
     * @param text
     * @param env
     * @return
     * @throws Exception
     */
    public Map<String, OnnxTensor> parseInputText(String text, OrtEnvironment env) throws Exception{
        var tokens = tokenizer.tokenize(text);
        var tokenIds = tokenizer.convert_tokens_to_ids(tokens);
        var inputIds = new long[tokenIds.size()];
        var attentionMask = new long[tokenIds.size()];
        var tokenTypeIds = new long[tokenIds.size()];
        for(var index=0; index < tokenIds.size(); index ++) {
            inputIds[index] = tokenIds.get(index);
            attentionMask[index] = 1;
            tokenTypeIds[index] = 0;
        }
        var shape = new long[]{1, inputIds.length};
        var ObjInputIds = OrtUtil.reshape(inputIds, shape);
        var ObjAttentionMask = OrtUtil.reshape(attentionMask, shape);
        var ObjTokenTypeIds = OrtUtil.reshape(tokenTypeIds, shape);
        var input_ids = OnnxTensor.createTensor(env, ObjInputIds);
        var attention_mask = OnnxTensor.createTensor(env, ObjAttentionMask);
        var token_type_ids = OnnxTensor.createTensor(env, ObjTokenTypeIds);
        var inputs = Map.of("input_ids", input_ids, "attention_mask", attention_mask, "token_type_ids", token_type_ids);
        return inputs;
    }

}
