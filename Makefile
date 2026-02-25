PYTHON ?= /usr/bin/python3

install:
	$(PYTHON) -m pip install -r requirements.txt

build-corpus:
	$(PYTHON) scripts/build_news_corpus.py --output_jsonl data/raw/news_10000.jsonl --num_samples 10000

auto-tag:
	$(PYTHON) scripts/auto_tag_gliner.py --model_name urchade/gliner_small-v2 --input_jsonl data/raw/news_10000.jsonl --output_jsonl data/interim/news_10000_tagged.jsonl --labels_config config/labels.json --threshold 0.60

validate-tagged:
	$(PYTHON) scripts/validate_tagged_data.py --tagged_jsonl data/interim/news_10000_tagged.jsonl --labels_config config/labels.json

convert:
	$(PYTHON) scripts/convert_to_gliner.py --input_jsonl data/interim/news_10000_tagged.jsonl --train_output data/processed/train_gliner.jsonl --valid_output data/processed/valid_gliner.jsonl --tokenizer_name urchade/gliner_small-v2 --valid_ratio 0.1

train:
	$(PYTHON) scripts/train_gliner.py --model_name urchade/gliner_small-v2 --train_jsonl data/processed/train_gliner.jsonl --valid_jsonl data/processed/valid_gliner.jsonl --output_dir outputs/run_001

pipeline:
	$(PYTHON) scripts/run_pipeline.py --model_name urchade/gliner_small-v2 --num_samples 10000
