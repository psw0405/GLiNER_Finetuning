# GLiNER ONNX Export

`outputs/run2/final` 모델을 ONNX로 변환해 `onnx/` 폴더에 저장합니다.

## 1) Export

```bash
python onnx/export_gliner_to_onnx.py \
  --model_dir outputs/run2/final \
  --output_dir onnx
```

기본 산출물:
- `onnx/model.onnx`
- `onnx/labels.json`
- `onnx/export_manifest.json`
- tokenizer 관련 파일 (`spm.model`, `tokenizer.json`, `tokenizer_config.json`, ...)

## 2) Tizen C++ 계획 검토

SentencePiece 사용 계획 자체는 가능합니다. 다만 다음을 반드시 맞춰야 합니다.

1. **Tokenizer parity**
   - 이 모델은 `spm.model` 외에 `added_tokens.json`, `special_tokens_map.json` 같은 추가 토큰 설정을 사용합니다.
   - C++ 토크나이저가 이 추가 토큰/정규화 규칙을 동일하게 재현하지 못하면 Python 결과와 달라질 수 있습니다.

2. **Pre/Post-processing parity**
   - GLiNER는 단순 `input_ids`만 쓰지 않고 `words_mask`, `text_lengths`, `span_idx`, `span_mask`까지 사용합니다.
   - C++에서 동일한 전처리(단어 분할 + span 인덱스 생성)와 후처리(logits -> entity decode)를 구현해야 같은 결과를 얻습니다.

3. **권장 방식**
   - ONNX Runtime(C++) + SentencePiece 조합은 가능하지만,
   - 초기 단계에서는 Python 추론 결과와 C++ 결과를 같은 입력으로 비교하면서 parity 테스트를 반드시 수행하세요.

## 3) 로딩 힌트 (Python)

GLiNER는 ONNX 로드 경로도 지원합니다.

```python
from gliner import GLiNER

model = GLiNER.from_pretrained(
    "onnx",
    load_onnx_model=True,
    onnx_model_file="model.onnx",
)
```

## 4) Tizen C++ 추론 코드

`onnx/cpp`에 다음이 구현되어 있습니다.

- `include/gliner_tizen_infer.hpp`
- `src/gliner_tizen_infer.cpp`
- `src/main.cpp` (샘플 CLI)
- `CMakeLists.txt`

구현 범위:

- 전처리 parity
   - whitespace 기반 단어 분할 (`\w+(?:[-_]\w+)*|\S` 동작 기준)
   - prompt 구성: `<<ENT>> label ... <<SEP>>`
   - `input_ids`, `attention_mask`, `words_mask`, `text_lengths`, `span_idx`, `span_mask` 생성
- 후처리 parity
   - `sigmoid(logits) > threshold`
   - span decode + greedy overlap suppression (`flat_ner`, `multi_label` 반영)
   - 최종 `start/end/text/label/score` 반환

### 빌드 예시

필수 의존성:
- ONNX Runtime C++
- SentencePiece C++
- nlohmann_json

Windows에서는 `onnx/cpp/build_windows.ps1` 사용을 권장합니다.

```powershell
powershell -ExecutionPolicy Bypass -File onnx/cpp/build_windows.ps1 \
   -OnnxRuntimeRoot C:/onnxruntime
```

이 스크립트는 다음을 자동 수행합니다.
- `.venv`의 `cmake` 자동 설치(없을 때)
- VS DevCmd 환경 import 시도
- `cl`/`nmake` 사전 점검

`cl`/`nmake`가 없으면 VC++ Build Tools 워크로드가 빠진 상태이므로, 관리자 권한 터미널에서 1회 설치가 필요합니다.

```powershell
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" modify \
   --installPath "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools" \
   --add Microsoft.VisualStudio.Workload.VCTools \
   --includeRecommended
```

```bash
cd onnx/cpp
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build --config Release
```

Windows에서 `onnxruntime.dll` 자동 복사를 원하면 CMake 옵션을 추가합니다.

```bash
cmake -S . -B build \
   -DONNXRUNTIME_ROOT=C:/onnxruntime \
   -DONNXRUNTIME_DLL=C:/onnxruntime/lib/onnxruntime.dll
```

### 실행 예시

```bash
./build/gliner_tizen_demo ../../onnx "삼성전자는 서울에서 신제품을 공개했다." 0.5 1 0
```

Zero-shot 라벨을 런타임에 넣고 싶다면 JSON 배열 파일을 마지막 인자로 전달할 수 있습니다.

예: `custom_labels.json`

```json
[
   "Company",
   "Product",
   "LaunchDate"
]
```

```bash
./build/gliner_tizen_demo ../../onnx "삼성전자는 서울에서 신제품을 공개했다." 0.5 1 0 ./custom_labels.json
```

주의: 현재 `model.onnx`가 고정 class 차원으로 export된 경우, `labels_json` 개수는 ONNX 출력 class 수와 같아야 합니다.

출력은 JSON 배열이며 각 항목은 `start`, `end`, `label`, `text`, `score`를 포함합니다.

## 5) Python vs C++ Parity 자동 비교

같은 입력/옵션으로 Python(GLiNER ONNX)과 C++(`gliner_tizen_demo`) 결과를 자동 비교합니다.

기본 실행(바이너리 자동 탐색):

```bash
python onnx/parity_test.py \
   --model_dir onnx \
   --threshold 0.5
```

텍스트 여러 개 비교:

```bash
python onnx/parity_test.py \
   --model_dir onnx \
   --text "삼성전자는 서울에서 신제품을 공개했다." \
   --text "홍길동은 부산에서 회의를 열었다."
```

바이너리 경로 직접 지정:

```bash
python onnx/parity_test.py \
   --model_dir onnx \
   --cpp_bin onnx/cpp/build/gliner_tizen_demo.exe
```

커스텀 라벨 파일로 parity 비교:

```bash
python onnx/parity_test.py \
   --model_dir onnx \
   --labels_file ./custom_labels.json \
   --text "삼성전자는 서울에서 신제품을 공개했다."
```

불일치 시 exit code 1로 종료하고, 누락 엔티티/score 차이를 출력합니다.
