# GLiNER_Finetuning

`urchade/gliner_small-v2` 기반 파인튜닝을 위한 데이터 생성/자동 라벨링/변환/학습 파이프라인입니다.

## 라벨 구성
- 최종 라벨 26개 사용 (`EventFestival` 제외)
- 라벨 소스: `config/labels.json`
- alias 지원:
	- `DateDuraion` -> `DateDuration`
	- `QunatityTemperature` -> `QuantityTemperature`

## 구성 파일
- `scripts/build_news_corpus.py`: 뉴스 문장 10,000개 생성(실제+합성 혼합)
- `scripts/auto_tag_gliner.py`: GLiNER self-labeling으로 span 자동 태깅
- `scripts/validate_tagged_data.py`: span/label/overlap 검증
- `scripts/convert_to_gliner.py`: GLiNER 학습 포맷(JSONL) 변환
- `scripts/train_gliner.py`: GLiNER 학습 API 호출 래퍼
- `scripts/run_pipeline.py`: 전체 파이프라인 실행
- `docs/model_analysis.md`: GLiNER small-v2 분석
- `docs/data_contract.md`: 데이터 포맷 계약

## 설치
```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

## 1) 10,000 뉴스 문장 생성
실제 뉴스 소스가 있으면 `--real_news_path`를 추가하세요.

```bash
/usr/bin/python3 scripts/build_news_corpus.py \
	--output_jsonl data/raw/news_10000.jsonl \
	--num_samples 10000 \
	--real_news_path path/to/real_news.jsonl \
	--real_ratio 0.6
```

`real_news.jsonl` 예시:
```json
{"text": "서울시가 오늘 신규 교통 정책을 발표했다."}
```

## 2) 자동 Span Tagging
```bash
/usr/bin/python3 scripts/auto_tag_gliner.py \
	--model_name urchade/gliner_small-v2 \
	--input_jsonl data/raw/news_10000.jsonl \
	--output_jsonl data/interim/news_10000_tagged.jsonl \
	--labels_config config/labels.json \
	--threshold 0.60
```

## 3) 태깅 검증
```bash
/usr/bin/python3 scripts/validate_tagged_data.py \
	--tagged_jsonl data/interim/news_10000_tagged.jsonl \
	--labels_config config/labels.json
```

## 4) GLiNER 학습 포맷 변환
```bash
/usr/bin/python3 scripts/convert_to_gliner.py \
	--input_jsonl data/interim/news_10000_tagged.jsonl \
	--train_output data/processed/train_gliner.jsonl \
	--valid_output data/processed/valid_gliner.jsonl \
	--tokenizer_name urchade/gliner_small-v2 \
	--valid_ratio 0.1
```

## 5) 파인튜닝 실행
```bash
/usr/bin/python3 scripts/train_gliner.py \
	--model_name urchade/gliner_small-v2 \
	--train_jsonl data/processed/train_gliner.jsonl \
	--valid_jsonl data/processed/valid_gliner.jsonl \
	--output_dir outputs/run_001
```

## 원클릭 파이프라인
```bash
/usr/bin/python3 scripts/run_pipeline.py \
	--model_name urchade/gliner_small-v2 \
	--num_samples 10000 \
	--real_news_path path/to/real_news.jsonl
```

## Makefile 사용
```bash
make install
make build-corpus
make auto-tag
make validate-tagged
make convert
make train
```

또는
```bash
make pipeline
```