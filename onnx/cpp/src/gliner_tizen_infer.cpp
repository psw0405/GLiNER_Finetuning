#include "gliner_tizen_infer.hpp"

#include <array>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace gliner_tizen {

namespace {

using json = nlohmann::json;

std::string ReadTextFile(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

json ReadJson(const std::filesystem::path& path) {
    return json::parse(ReadTextFile(path));
}

void ValidateLabels(const std::vector<std::string>& labels, const std::string& context) {
    if (labels.empty()) {
        throw std::runtime_error(context + " labels must not be empty.");
    }

    std::unordered_set<std::string> seen;
    seen.reserve(labels.size());

    for (const auto& label : labels) {
        if (label.empty()) {
            throw std::runtime_error(context + " labels contain an empty string.");
        }
        if (!seen.insert(label).second) {
            throw std::runtime_error(context + " labels contain duplicate entry: " + label);
        }
    }
}

}

GlinerOnnxInfer::GlinerOnnxInfer(const std::string& model_dir, const std::string& onnx_file)
    : model_dir_(model_dir),
      env_(ORT_LOGGING_LEVEL_WARNING, "gliner_tizen"),
      session_options_() {
    LoadAssets(model_dir);
    InitSession(model_dir, onnx_file);
    ValidateInputSchema();
}

std::vector<Entity> GlinerOnnxInfer::Predict(const std::string& text, const InferOptions& options) const {
    auto batch = PredictBatch(std::vector<std::string>{text}, labels_, options);
    if (batch.empty()) {
        return {};
    }
    return std::move(batch[0]);
}

std::vector<Entity> GlinerOnnxInfer::Predict(
    const std::string& text,
    const std::vector<std::string>& labels,
    const InferOptions& options
) const {
    auto batch = PredictBatch(std::vector<std::string>{text}, labels, options);
    if (batch.empty()) {
        return {};
    }
    return std::move(batch[0]);
}

std::vector<std::vector<Entity>> GlinerOnnxInfer::PredictBatch(
    const std::vector<std::string>& texts,
    const InferOptions& options
) const {
    return PredictBatch(texts, labels_, options);
}

std::vector<std::vector<Entity>> GlinerOnnxInfer::PredictBatch(
    const std::vector<std::string>& texts,
    const std::vector<std::string>& labels,
    const InferOptions& options
) const {
    if (texts.empty()) {
        return {};
    }

    ValidateLabels(labels, "PredictBatch");

    std::vector<PreparedExample> prepared;
    prepared.reserve(texts.size());
    int64_t max_seq_len = 0;
    int64_t max_spans = 0;

    for (const auto& text : texts) {
        PreparedExample ex = PrepareExample(text, labels);
        max_seq_len = std::max<int64_t>(max_seq_len, static_cast<int64_t>(ex.input_ids.size()));
        max_spans = std::max<int64_t>(max_spans, static_cast<int64_t>(ex.span_mask.size()));
        prepared.push_back(std::move(ex));
    }

    const int64_t batch_size = static_cast<int64_t>(prepared.size());

    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> words_mask;
    std::vector<int64_t> text_lengths;
    std::vector<int64_t> span_idx;
    std::vector<uint8_t> span_mask_u8;

    input_ids.reserve(static_cast<size_t>(batch_size * max_seq_len));
    attention_mask.reserve(static_cast<size_t>(batch_size * max_seq_len));
    words_mask.reserve(static_cast<size_t>(batch_size * max_seq_len));
    text_lengths.reserve(static_cast<size_t>(batch_size));
    span_idx.reserve(static_cast<size_t>(batch_size * max_spans * 2));
    span_mask_u8.reserve(static_cast<size_t>(batch_size * max_spans));

    for (const auto& ex : prepared) {
        for (int64_t i = 0; i < max_seq_len; ++i) {
            if (i < static_cast<int64_t>(ex.input_ids.size())) {
                input_ids.push_back(ex.input_ids[static_cast<size_t>(i)]);
                attention_mask.push_back(ex.attention_mask[static_cast<size_t>(i)]);
                words_mask.push_back(ex.words_mask[static_cast<size_t>(i)]);
            } else {
                input_ids.push_back(pad_id_);
                attention_mask.push_back(0);
                words_mask.push_back(0);
            }
        }

        text_lengths.push_back(ex.text_length);

        const int64_t ex_spans = static_cast<int64_t>(ex.span_mask.size());
        for (int64_t i = 0; i < max_spans; ++i) {
            if (i < ex_spans) {
                span_idx.push_back(ex.span_idx[static_cast<size_t>(2 * i)]);
                span_idx.push_back(ex.span_idx[static_cast<size_t>(2 * i + 1)]);
                span_mask_u8.push_back(ex.span_mask[static_cast<size_t>(i)] ? 1 : 0);
            } else {
                span_idx.push_back(0);
                span_idx.push_back(0);
                span_mask_u8.push_back(0);
            }
        }
    }

    std::unique_ptr<bool[]> span_mask_bool = std::make_unique<bool[]>(span_mask_u8.size());
    for (size_t i = 0; i < span_mask_u8.size(); ++i) {
        span_mask_bool[i] = (span_mask_u8[i] != 0);
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> seq_shape{batch_size, max_seq_len};
    std::array<int64_t, 2> text_length_shape{batch_size, 1};
    std::array<int64_t, 3> span_idx_shape{batch_size, max_spans, 2};
    std::array<int64_t, 2> span_mask_shape{batch_size, max_spans};

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(6);
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, input_ids.data(), input_ids.size(), seq_shape.data(), seq_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, attention_mask.data(), attention_mask.size(), seq_shape.data(), seq_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, words_mask.data(), words_mask.size(), seq_shape.data(), seq_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, text_lengths.data(), text_lengths.size(), text_length_shape.data(), text_length_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, span_idx.data(), span_idx.size(), span_idx_shape.data(), span_idx_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<bool>(mem_info, span_mask_bool.get(), span_mask_u8.size(), span_mask_shape.data(), span_mask_shape.size()));

    std::array<const char*, 6> input_names = {
        "input_ids",
        "attention_mask",
        "words_mask",
        "text_lengths",
        "span_idx",
        "span_mask"
    };

    const char* output_name = "logits";

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        &output_name,
        1
    );

    if (outputs.empty()) {
        throw std::runtime_error("ONNX inference returned no outputs.");
    }

    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4) {
        throw std::runtime_error("Unexpected logits rank. Expected 4D tensor.");
    }

    const int64_t out_batch = shape[0];
    const int64_t out_words = shape[1];
    const int64_t out_max_width = shape[2];
    const int64_t out_num_classes = shape[3];

    if (out_batch != batch_size) {
        throw std::runtime_error("Output batch size mismatch.");
    }
    if (out_num_classes != static_cast<int64_t>(labels.size())) {
        throw std::runtime_error(
            "Output class dimension mismatch. "
            "logits classes=" + std::to_string(out_num_classes)
            + ", requested labels=" + std::to_string(labels.size())
            + ". Re-export ONNX for dynamic labels or pass matching label count."
        );
    }

    const float* logits = outputs[0].GetTensorData<float>();

    std::vector<std::vector<Entity>> all_entities;
    all_entities.reserve(prepared.size());

    const int64_t stride_batch = out_words * out_max_width * out_num_classes;

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_logits = logits + b * stride_batch;
        auto entities = DecodeSpans(
            prepared[static_cast<size_t>(b)],
            batch_logits,
            out_words,
            out_max_width,
            out_num_classes,
            labels,
            options
        );
        all_entities.push_back(std::move(entities));
    }

    return all_entities;
}

void GlinerOnnxInfer::LoadAssets(const std::string& model_dir) {
    namespace fs = std::filesystem;
    const fs::path dir(model_dir);

    const fs::path labels_path = dir / "labels.json";
    const fs::path spm_path = dir / "spm.model";
    const fs::path tokenizer_cfg_path = dir / "tokenizer_config.json";
    const fs::path added_tokens_path = dir / "added_tokens.json";
    const fs::path gliner_cfg_path = dir / "gliner_config.json";

    auto labels_json = ReadJson(labels_path);
    labels_.clear();
    for (const auto& item : labels_json) {
        labels_.push_back(item.get<std::string>());
    }
    ValidateLabels(labels_, "labels.json");

    if (!sp_.Load(spm_path.string()).ok()) {
        throw std::runtime_error("Failed to load sentencepiece model: " + spm_path.string());
    }

    auto tokenizer_cfg = ReadJson(tokenizer_cfg_path);
    if (tokenizer_cfg.contains("added_tokens_decoder")) {
        const auto& decoder = tokenizer_cfg["added_tokens_decoder"];
        for (auto it = decoder.begin(); it != decoder.end(); ++it) {
            int64_t id = std::stoll(it.key());
            const auto& obj = it.value();
            if (obj.contains("content")) {
                token_to_id_[obj["content"].get<std::string>()] = id;
            }
        }
    }

    if (token_to_id_.count("[PAD]")) pad_id_ = token_to_id_.at("[PAD]");
    if (token_to_id_.count("[CLS]")) cls_id_ = token_to_id_.at("[CLS]");
    if (token_to_id_.count("[SEP]")) sep_id_ = token_to_id_.at("[SEP]");
    if (token_to_id_.count("[UNK]")) unk_id_ = token_to_id_.at("[UNK]");

    auto added_tokens_json = ReadJson(added_tokens_path);
    for (auto it = added_tokens_json.begin(); it != added_tokens_json.end(); ++it) {
        token_to_id_[it.key()] = it.value().get<int64_t>();
    }

    auto gliner_cfg = ReadJson(gliner_cfg_path);
    if (gliner_cfg.contains("max_width")) {
        max_width_ = gliner_cfg["max_width"].get<int64_t>();
    }
    if (gliner_cfg.contains("max_len")) {
        max_len_ = gliner_cfg["max_len"].get<int64_t>();
    }
    if (gliner_cfg.contains("ent_token")) {
        ent_token_ = gliner_cfg["ent_token"].get<std::string>();
    }
    if (gliner_cfg.contains("sep_token")) {
        sep_prompt_token_ = gliner_cfg["sep_token"].get<std::string>();
    }

    if (!token_to_id_.count(ent_token_)) {
        throw std::runtime_error("ent_token id is missing in tokenizer assets: " + ent_token_);
    }
    if (!token_to_id_.count(sep_prompt_token_)) {
        throw std::runtime_error("sep_prompt token id is missing in tokenizer assets: " + sep_prompt_token_);
    }
}

void GlinerOnnxInfer::InitSession(const std::string& model_dir, const std::string& onnx_file) {
    namespace fs = std::filesystem;
    const fs::path onnx_path = fs::path(model_dir) / onnx_file;

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(1);

#ifdef _WIN32
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.wstring().c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, onnx_path.string().c_str(), session_options_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    input_names_.clear();
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name.get());
    }

    output_names_.clear();
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
    }
}

void GlinerOnnxInfer::ValidateInputSchema() const {
    static const std::set<std::string> required = {
        "input_ids",
        "attention_mask",
        "words_mask",
        "text_lengths",
        "span_idx",
        "span_mask"
    };

    std::set<std::string> actual(input_names_.begin(), input_names_.end());
    for (const auto& name : required) {
        if (!actual.count(name)) {
            throw std::runtime_error("ONNX input is missing: " + name);
        }
    }

    if (std::find(output_names_.begin(), output_names_.end(), "logits") == output_names_.end()) {
        throw std::runtime_error("ONNX output 'logits' is missing.");
    }
}

std::vector<GlinerOnnxInfer::Utf8Codepoint> GlinerOnnxInfer::DecodeUtf8(
    const std::string& text,
    std::vector<size_t>& char_to_byte
) const {
    std::vector<Utf8Codepoint> out;
    out.reserve(text.size());

    size_t i = 0;
    int32_t char_idx = 0;

    while (i < text.size()) {
        const unsigned char c0 = static_cast<unsigned char>(text[i]);
        size_t len = 1;
        uint32_t cp = 0xFFFD;

        if ((c0 & 0x80u) == 0u) {
            cp = c0;
            len = 1;
        } else if ((c0 & 0xE0u) == 0xC0u && i + 1 < text.size()) {
            cp = ((c0 & 0x1Fu) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3Fu);
            len = 2;
        } else if ((c0 & 0xF0u) == 0xE0u && i + 2 < text.size()) {
            cp = ((c0 & 0x0Fu) << 12)
                | ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 6)
                | (static_cast<unsigned char>(text[i + 2]) & 0x3Fu);
            len = 3;
        } else if ((c0 & 0xF8u) == 0xF0u && i + 3 < text.size()) {
            cp = ((c0 & 0x07u) << 18)
                | ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 12)
                | ((static_cast<unsigned char>(text[i + 2]) & 0x3Fu) << 6)
                | (static_cast<unsigned char>(text[i + 3]) & 0x3Fu);
            len = 4;
        }

        out.push_back(Utf8Codepoint{cp, i, i + len, char_idx});
        i += len;
        char_idx += 1;
    }

    char_to_byte.assign(static_cast<size_t>(char_idx + 1), text.size());
    for (const auto& cp : out) {
        char_to_byte[static_cast<size_t>(cp.char_index)] = cp.byte_start;
    }
    char_to_byte[static_cast<size_t>(char_idx)] = text.size();
    return out;
}

std::vector<GlinerOnnxInfer::TokenSpan> GlinerOnnxInfer::SplitWords(
    const std::string& text,
    const std::vector<Utf8Codepoint>& cps
) const {
    std::vector<TokenSpan> tokens;
    tokens.reserve(cps.size());

    size_t i = 0;
    while (i < cps.size()) {
        if (IsWhitespace(cps[i].cp)) {
            i += 1;
            continue;
        }

        if (IsWordChar(cps[i].cp)) {
            const size_t start = i;
            size_t j = i;

            while (j < cps.size() && IsWordChar(cps[j].cp)) {
                j += 1;
            }

            while (
                j + 1 < cps.size()
                && (cps[j].cp == static_cast<uint32_t>('-') || cps[j].cp == static_cast<uint32_t>('_'))
                && IsWordChar(cps[j + 1].cp)
            ) {
                j += 1;
                while (j < cps.size() && IsWordChar(cps[j].cp)) {
                    j += 1;
                }
            }

            const size_t byte_start = cps[start].byte_start;
            const size_t byte_end = cps[j - 1].byte_end;

            tokens.push_back(TokenSpan{
                text.substr(byte_start, byte_end - byte_start),
                cps[start].char_index,
                cps[j - 1].char_index + 1,
            });

            i = j;
            continue;
        }

        tokens.push_back(TokenSpan{
            text.substr(cps[i].byte_start, cps[i].byte_end - cps[i].byte_start),
            cps[i].char_index,
            cps[i].char_index + 1,
        });

        i += 1;
    }

    if (static_cast<int64_t>(tokens.size()) > max_len_) {
        tokens.resize(static_cast<size_t>(max_len_));
    }

    return tokens;
}

std::vector<int64_t> GlinerOnnxInfer::EncodeWord(const std::string& word) const {
    const auto it = token_to_id_.find(word);
    if (it != token_to_id_.end()) {
        return std::vector<int64_t>{it->second};
    }

    std::vector<int> ids;
    const auto status = sp_.Encode(word, &ids);
    if (!status.ok() || ids.empty()) {
        return std::vector<int64_t>{unk_id_};
    }

    std::vector<int64_t> out;
    out.reserve(ids.size());
    for (int id : ids) {
        out.push_back(static_cast<int64_t>(id));
    }
    return out;
}

GlinerOnnxInfer::PreparedExample GlinerOnnxInfer::PrepareExample(
    const std::string& text,
    const std::vector<std::string>& labels
) const {
    PreparedExample ex;
    ex.text = text;

    auto cps = DecodeUtf8(text, ex.char_to_byte);
    ex.tokens = SplitWords(text, cps);
    if (ex.tokens.empty()) {
        ex.tokens.push_back(TokenSpan{"[PAD]", 0, 0});
    }

    std::vector<std::string> words;
    words.reserve(labels.size() * 2 + 1 + ex.tokens.size());

    for (const auto& label : labels) {
        words.push_back(ent_token_);
        words.push_back(label);
    }
    words.push_back(sep_prompt_token_);

    const int64_t prompt_length = static_cast<int64_t>(words.size());

    for (const auto& t : ex.tokens) {
        words.push_back(t.token);
    }

    ex.input_ids.clear();
    std::vector<int64_t> word_ids;

    ex.input_ids.push_back(cls_id_);
    word_ids.push_back(-1);

    for (size_t word_index = 0; word_index < words.size(); ++word_index) {
        auto ids = EncodeWord(words[word_index]);
        for (auto id : ids) {
            ex.input_ids.push_back(id);
            word_ids.push_back(static_cast<int64_t>(word_index));
        }
    }

    ex.input_ids.push_back(sep_id_);
    word_ids.push_back(-1);

    ex.attention_mask.assign(ex.input_ids.size(), 1);
    ex.words_mask.assign(ex.input_ids.size(), 0);

    int64_t prev_word_id = -10;
    int64_t words_count = 0;

    for (size_t i = 0; i < word_ids.size(); ++i) {
        const int64_t word_id = word_ids[i];

        if (word_id < 0) {
            ex.words_mask[i] = 0;
            prev_word_id = word_id;
            continue;
        }

        if (word_id != prev_word_id) {
            if (words_count < prompt_length) {
                ex.words_mask[i] = 0;
            } else {
                ex.words_mask[i] = word_id - prompt_length + 1;
            }
            words_count += 1;
        } else {
            ex.words_mask[i] = 0;
        }

        prev_word_id = word_id;
    }

    ex.text_length = static_cast<int64_t>(ex.tokens.size());
    ex.span_idx.reserve(static_cast<size_t>(ex.text_length * max_width_ * 2));
    ex.span_mask.reserve(static_cast<size_t>(ex.text_length * max_width_));

    for (int64_t start = 0; start < ex.text_length; ++start) {
        for (int64_t width = 0; width < max_width_; ++width) {
            const int64_t end = start + width;
            ex.span_idx.push_back(start);
            ex.span_idx.push_back(end);
            ex.span_mask.push_back(end <= (ex.text_length - 1) ? 1 : 0);
        }
    }

    return ex;
}

std::vector<Entity> GlinerOnnxInfer::DecodeSpans(
    const PreparedExample& ex,
    const float* logits,
    int64_t words_dim,
    int64_t max_width_dim,
    int64_t num_classes_dim,
    const std::vector<std::string>& labels,
    const InferOptions& options
) const {
    std::vector<SpanCandidate> candidates;

    const int64_t token_count = static_cast<int64_t>(ex.tokens.size());

    for (int64_t s = 0; s < words_dim; ++s) {
        for (int64_t k = 0; k < max_width_dim; ++k) {
            const int64_t end_exclusive = s + k + 1;
            if (end_exclusive > token_count) {
                continue;
            }

            for (int64_t c = 0; c < num_classes_dim; ++c) {
                const int64_t offset = ((s * max_width_dim) + k) * num_classes_dim + c;
                const float prob = Sigmoid(logits[offset]);
                if (prob > options.threshold) {
                    candidates.push_back(SpanCandidate{
                        static_cast<int32_t>(s),
                        static_cast<int32_t>(s + k),
                        static_cast<int32_t>(c),
                        prob,
                    });
                }
            }
        }
    }

    auto selected = GreedySearch(std::move(candidates), options.flat_ner, options.multi_label);

    std::vector<Entity> entities;
    entities.reserve(selected.size());

    for (const auto& span : selected) {
        if (span.start < 0 || span.end < span.start || span.end >= static_cast<int32_t>(ex.tokens.size())) {
            continue;
        }
        if (span.label_index < 0 || span.label_index >= static_cast<int32_t>(labels.size())) {
            continue;
        }

        const int32_t start_char = ex.tokens[static_cast<size_t>(span.start)].start_char;
        const int32_t end_char = ex.tokens[static_cast<size_t>(span.end)].end_char;

        if (start_char < 0 || end_char < start_char || end_char >= static_cast<int32_t>(ex.char_to_byte.size())) {
            continue;
        }

        const size_t start_byte = ex.char_to_byte[static_cast<size_t>(start_char)];
        const size_t end_byte = ex.char_to_byte[static_cast<size_t>(end_char)];
        if (end_byte < start_byte || end_byte > ex.text.size()) {
            continue;
        }

        entities.push_back(Entity{
            start_char,
            end_char,
            labels[static_cast<size_t>(span.label_index)],
            ex.text.substr(start_byte, end_byte - start_byte),
            span.score,
        });
    }

    return entities;
}

std::vector<GlinerOnnxInfer::SpanCandidate> GlinerOnnxInfer::GreedySearch(
    std::vector<SpanCandidate> spans,
    bool flat_ner,
    bool multi_label
) const {
    std::sort(spans.begin(), spans.end(), [](const SpanCandidate& a, const SpanCandidate& b) {
        return a.score > b.score;
    });

    std::vector<SpanCandidate> kept;
    kept.reserve(spans.size());

    for (const auto& cand : spans) {
        bool overlap = false;
        for (const auto& acc : kept) {
            const bool has_overlap = flat_ner
                ? HasOverlapping(cand, acc, multi_label)
                : HasOverlappingNested(cand, acc, multi_label);
            if (has_overlap) {
                overlap = true;
                break;
            }
        }
        if (!overlap) {
            kept.push_back(cand);
        }
    }

    std::sort(kept.begin(), kept.end(), [](const SpanCandidate& a, const SpanCandidate& b) {
        return a.start < b.start;
    });

    return kept;
}

bool GlinerOnnxInfer::HasOverlapping(const SpanCandidate& a, const SpanCandidate& b, bool multi_label) const {
    if (a.start == b.start && a.end == b.end) {
        return !multi_label;
    }
    if (a.start > b.end || b.start > a.end) {
        return false;
    }
    return true;
}

bool GlinerOnnxInfer::HasOverlappingNested(const SpanCandidate& a, const SpanCandidate& b, bool multi_label) const {
    if (a.start == b.start && a.end == b.end) {
        return !multi_label;
    }

    const bool disjoint = (a.start > b.end || b.start > a.end);
    if (disjoint) {
        return false;
    }

    const bool nested =
        ((a.start <= b.start) && (a.end >= b.end))
        || ((b.start <= a.start) && (b.end >= a.end));

    if (nested) {
        return false;
    }

    return true;
}

float GlinerOnnxInfer::Sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

bool GlinerOnnxInfer::IsWhitespace(uint32_t cp) {
    if (cp <= 0x20u) {
        return true;
    }
    return cp == 0x85u || cp == 0xA0u || cp == 0x1680u || (cp >= 0x2000u && cp <= 0x200Au)
        || cp == 0x2028u || cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

bool GlinerOnnxInfer::IsPunctuation(uint32_t cp) {
    if (cp < 128u) {
        return std::ispunct(static_cast<unsigned char>(cp)) != 0;
    }

    if ((cp >= 0x2000u && cp <= 0x206Fu)
        || (cp >= 0x2E00u && cp <= 0x2E7Fu)
        || (cp >= 0x3000u && cp <= 0x303Fu)
        || (cp >= 0xFE10u && cp <= 0xFE1Fu)
        || (cp >= 0xFE30u && cp <= 0xFE4Fu)
        || (cp >= 0xFF00u && cp <= 0xFF65u)) {
        return true;
    }

    return false;
}

bool GlinerOnnxInfer::IsWordChar(uint32_t cp) {
    if (cp == static_cast<uint32_t>('_')) {
        return true;
    }

    if (cp < 128u) {
        return std::isalnum(static_cast<unsigned char>(cp)) != 0;
    }

    if (IsWhitespace(cp) || IsPunctuation(cp)) {
        return false;
    }

    return true;
}

}
