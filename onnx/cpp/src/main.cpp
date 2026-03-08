#include "gliner_tizen_infer.hpp"

#include <exception>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

using gliner_tizen::Entity;
using gliner_tizen::GlinerOnnxInfer;
using gliner_tizen::InferOptions;

namespace {

void PrintUsage() {
    std::cerr
        << "Usage:\n"
        << "  gliner_tizen_demo <model_dir> <text> [threshold] [flat_ner] [multi_label]\n\n"
        << "Args:\n"
        << "  model_dir   Directory containing model.onnx, labels.json, spm.model, tokenizer files\n"
        << "  text        Input text (UTF-8)\n"
        << "  threshold   Optional score threshold (default: 0.5)\n"
        << "  flat_ner    Optional 1/0 (default: 1)\n"
        << "  multi_label Optional 1/0 (default: 0)\n";
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
        std::vector<Entity> entities = infer.Predict(text, options);

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
