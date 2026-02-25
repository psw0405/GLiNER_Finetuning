# 데이터 계약

## 1) 원천 코퍼스 (`data/raw/news_10000.jsonl`)
JSONL 한 줄 예시:

```json
{"id": 1, "text": "2026년 2월 3일 서울에서 열린 정상회담에서 김민수는 삼성전자의 입장을 발표했다.", "source": "synthetic"}
```

필수 필드:
- `id`: 정수
- `text`: 문자열
- `source`: `real` 또는 `synthetic`

## 2) 자동 태깅 결과 (`data/interim/news_10000_tagged.jsonl`)

```json
{"id": 1, "text": "...", "entities": [{"start": 0, "end": 10, "label": "Date", "text": "2026년 2월 3일", "score": 0.87}], "source": "synthetic"}
```

필수 규칙:
- `0 <= start < end <= len(text)`
- `label`은 `config/labels.json`의 labels에 포함되어야 함
- 오탈자 라벨은 alias로 정규화됨

## 3) GLiNER 학습 포맷
- train: `data/processed/train_gliner.jsonl`
- valid: `data/processed/valid_gliner.jsonl`

```json
{"tokenized_text": ["2026", "년", "2", "월"], "ner": [[0, 3, "Date"], [8, 9, "LocationCity"]]}
```

필수 규칙:
- `ner`의 start/end는 토큰 인덱스
- 동일 span/label 중복 제거
- 빈 토큰 샘플 제외
