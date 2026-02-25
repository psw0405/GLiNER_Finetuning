# GLiNER small-v2 분석

## 모델 개요
- 베이스 모델: `urchade/gliner_small-v2`
- 태스크: 범용 NER(라벨 목록을 추론 시점에 동적으로 전달)
- 핵심 API: `GLiNER.from_pretrained(...)`, `model.predict_entities(text, labels=[...], threshold=...)`

## 입력/출력
- 입력 텍스트: 단일 문자열
- 라벨 리스트: 이번 프로젝트는 `config/labels.json`의 26개 라벨 고정
- 출력 엔티티: `start`, `end`, `label`, `text`, `score`

## 파인튜닝 데이터 요구사항
- 최종 학습 포맷은 `tokenized_text` + `ner`
- `tokenized_text`: 토큰 문자열 배열
- `ner`: `[start_token_idx, end_token_idx, label]` 배열(포함 범위)

## 현재 파이프라인 적용 방식
1. 뉴스 문장 10,000개 확보(실제+합성 혼합)
2. GLiNER self-labeling으로 char span 추출
3. 겹침 제거/라벨 alias 정규화
4. 토큰 인덱스로 변환 후 train/valid 분할
