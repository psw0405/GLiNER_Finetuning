#include "gliner_tizen_infer.hpp"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using gliner_tizen::Entity;
using gliner_tizen::GlinerOnnxInfer;
using gliner_tizen::InferOptions;

namespace {

void PrintUsage() {
    std::cerr
        << "Usage:\n"
        << "  gliner_tizen_demo <model_dir> <text> [threshold] [flat_ner] [multi_label] [labels_json]\n\n"
        << "Args:\n"
        << "  model_dir   Directory containing model.onnx, labels.json, spm.model, tokenizer files\n"
        << "  text        Input text (UTF-8)\n"
        << "  threshold   Optional score threshold (default: 0.5)\n"
        << "  flat_ner    Optional 1/0 (default: 1)\n"
        << "  multi_label Optional 1/0 (default: 0)\n"
        << "  labels_json Optional path to zero-shot labels JSON array\n";
}

std::vector<std::string> LoadLabelsFromFile(const std::string& labels_path) {
    std::ifstream ifs(labels_path, std::ios::in | std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open labels file: " + labels_path);
    }

    const nlohmann::json parsed = nlohmann::json::parse(ifs);
    if (!parsed.is_array()) {
        throw std::runtime_error("labels_json must be a JSON array of strings: " + labels_path);
    }

    std::vector<std::string> labels;
    labels.reserve(parsed.size());

    for (const auto& item : parsed) {
        if (!item.is_string()) {
            throw std::runtime_error("labels_json must contain only strings: " + labels_path);
        }
        const std::string label = item.get<std::string>();
        if (label.empty()) {
            throw std::runtime_error("labels_json contains an empty label: " + labels_path);
        }
        labels.push_back(label);
    }

    if (labels.empty()) {
        throw std::runtime_error("labels_json must not be empty: " + labels_path);
    }

    return labels;
}

}

int main(int argc, char** argv) {
    if (argc < 3) {
        PrintUsage();
        return 1;
    }

    const std::string model_dir = argv[1];
    const std::string text = argv[2];

    InferOptions options;
    if (argc >= 4) {
        options.threshold = std::stof(argv[3]);
    }
    if (argc >= 5) {
        options.flat_ner = (std::stoi(argv[4]) != 0);
    }
    if (argc >= 6) {
        options.multi_label = (std::stoi(argv[5]) != 0);
    }

    try {
        GlinerOnnxInfer infer(model_dir, "model.onnx");
        std::vector<Entity> entities;
        if (argc >= 7) {
            std::vector<std::string> labels = LoadLabelsFromFile(argv[6]);
            entities = infer.Predict(text, labels, options);
        } else {
            entities = infer.Predict(text, options);
        }

        nlohmann::json out = nlohmann::json::array();
        for (const auto& e : entities) {
            out.push_back(
                {
                    {"start", e.start},
                    {"end", e.end},
                    {"label", e.label},
                    {"text", e.text},
                    {"score", e.score}
                }
            );
        }

        std::cout << out.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Inference error: " << ex.what() << std::endl;
        return 2;
    }
}
