#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>

namespace gliner_tizen {

struct Entity {
    int32_t start;
    int32_t end;
    std::string label;
    std::string text;
    float score;
};

struct InferOptions {
    float threshold = 0.5f;
    bool flat_ner = true;
    bool multi_label = false;
};

class GlinerOnnxInfer {
public:
    explicit GlinerOnnxInfer(const std::string& model_dir, const std::string& onnx_file = "model.onnx");
    std::vector<Entity> Predict(const std::string& text, const InferOptions& options = InferOptions()) const;
    std::vector<Entity> Predict(
        const std::string& text,
        const std::vector<std::string>& labels,
        const InferOptions& options = InferOptions()
    ) const;
    std::vector<std::vector<Entity>> PredictBatch(const std::vector<std::string>& texts, const InferOptions& options = InferOptions()) const;
    std::vector<std::vector<Entity>> PredictBatch(
        const std::vector<std::string>& texts,
        const std::vector<std::string>& labels,
        const InferOptions& options = InferOptions()
    ) const;

private:
    struct TokenSpan {
        std::string token;
        int32_t start_char;
        int32_t end_char;
    };

    struct PreparedExample {
        std::string text;
        std::vector<TokenSpan> tokens;
        std::vector<size_t> char_to_byte;
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
        std::vector<int64_t> words_mask;
        std::vector<int64_t> span_idx;
        std::vector<uint8_t> span_mask;
        int64_t text_length = 0;
    };

    struct SpanCandidate {
        int32_t start;
        int32_t end;
        int32_t label_index;
        float score;
    };

    struct Utf8Codepoint {
        uint32_t cp;
        size_t byte_start;
        size_t byte_end;
        int32_t char_index;
    };

    void LoadAssets(const std::string& model_dir);
    void InitSession(const std::string& model_dir, const std::string& onnx_file);
    void ValidateInputSchema() const;

    std::vector<Utf8Codepoint> DecodeUtf8(const std::string& text, std::vector<size_t>& char_to_byte) const;
    std::vector<TokenSpan> SplitWords(const std::string& text, const std::vector<Utf8Codepoint>& cps) const;
    std::vector<int64_t> EncodeWord(const std::string& word) const;

    PreparedExample PrepareExample(const std::string& text, const std::vector<std::string>& labels) const;
    std::vector<Entity> DecodeSpans(
        const PreparedExample& ex,
        const float* logits,
        int64_t words_dim,
        int64_t max_width_dim,
        int64_t num_classes_dim,
        const std::vector<std::string>& labels,
        const InferOptions& options
    ) const;

    std::vector<SpanCandidate> GreedySearch(std::vector<SpanCandidate> spans, bool flat_ner, bool multi_label) const;
    bool HasOverlapping(const SpanCandidate& a, const SpanCandidate& b, bool multi_label) const;
    bool HasOverlappingNested(const SpanCandidate& a, const SpanCandidate& b, bool multi_label) const;

    static float Sigmoid(float x);

    static bool IsWhitespace(uint32_t cp);
    static bool IsPunctuation(uint32_t cp);
    static bool IsWordChar(uint32_t cp);

    std::string model_dir_;

    std::vector<std::string> labels_;
    std::unordered_map<std::string, int64_t> token_to_id_;

    int64_t pad_id_ = 0;
    int64_t cls_id_ = 1;
    int64_t sep_id_ = 2;
    int64_t unk_id_ = 3;

    std::string ent_token_ = "<<ENT>>";
    std::string sep_prompt_token_ = "<<SEP>>";

    int64_t max_width_ = 12;
    int64_t max_len_ = 384;

    sentencepiece::SentencePieceProcessor sp_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
};

}  // namespace gliner_tizen
