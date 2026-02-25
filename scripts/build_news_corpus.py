import argparse
import json
import random
from pathlib import Path
from typing import Iterable


PERSONS = ["김민수", "이서연", "박지훈", "정하늘", "최유진", "손흥민", "김연아", "오타니 쇼헤이"]
COUNTRIES = ["대한민국", "일본", "미국", "프랑스", "독일", "브라질", "캐나다"]
CITIES = ["서울", "부산", "인천", "도쿄", "뉴욕", "파리", "베를린"]
ORGS = ["삼성전자", "현대자동차", "네이버", "카카오", "LG화학", "대한축구협회"]
SHOPS = ["롯데백화점", "이마트", "쿠팡", "올리브영", "스타벅스"]
BUILDINGS = ["롯데월드타워", "국회의사당", "서울시청", "코엑스", "부산국제금융센터"]
CULTURE_SITES = ["경복궁", "덕수궁", "창덕궁", "불국사", "수원화성"]
SPORTS = ["축구", "야구", "농구", "배구", "테니스", "수영"]
FOODS = ["김치", "비빔밥", "불고기", "라면", "떡볶이"]
MED_TERMS = ["독감", "폐렴", "고혈압", "당뇨", "코로나19", "천식"]
CURRENCY = ["원", "달러", "유로", "엔"]
LAW = ["근로기준법", "상법", "형법", "민법", "도로교통법"]
ANIMALS = ["호랑이", "고양이", "강아지", "판다", "독수리"]
EVENTS = ["정상회담", "기자회견", "개막식", "폐막식", "국정감사"]
EVENT_SPORTS = ["아시안게임", "월드컵", "올림픽", "K리그", "프로야구 개막전"]


TEMPLATES = [
    "{date} {city}에서 열린 {event}에서 {person}은 {organization}의 입장을 발표했다.",
    "{organization}는 {date} {country} {city}에 신규 {building} 건설 계획을 공개했다.",
    "{person}은 {sports} 경기 도중 {time_duration} 동안 치료를 받고 복귀했다.",
    "{shop}는 {date}부터 {food} 가격을 {price} 인상한다고 밝혔다.",
    "{culture_site} 인근에서 발견된 {animal} 관련 소식이 {city} 시민들 사이에서 화제가 됐다.",
    "{organization}는 {term_medical} 대응 지침을 {law} 기준에 맞춰 개정했다.",
    "{event_sports} 조직위원회는 {date_duration} 준비 기간 동안 {quantity}명의 자원봉사자를 모집했다.",
    "{person}은 {country} 통화인 {currency} 환율 급등으로 수입 비용이 증가했다고 말했다.",
    "{city} 기온이 {temperature}까지 내려가며 시민들은 외출 시간을 {time} 이후로 미뤘다.",
    "{organization}는 {date} 기준 {age} 연령층 대상 교육 프로그램을 확대했다."
]


def read_real_sentences(path: Path) -> list[str]:
    sentences: list[str] = []
    if not path.exists():
        return sentences
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text", "").strip()
                if text:
                    sentences.append(text)
    else:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    sentences.append(text)
    return sentences


def make_context() -> dict[str, str]:
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    hour = random.randint(7, 22)
    minute = random.choice([0, 10, 20, 30, 40, 50])
    return {
        "person": random.choice(PERSONS),
        "country": random.choice(COUNTRIES),
        "city": random.choice(CITIES),
        "organization": random.choice(ORGS),
        "shop": random.choice(SHOPS),
        "building": random.choice(BUILDINGS),
        "culture_site": random.choice(CULTURE_SITES),
        "sports": random.choice(SPORTS),
        "food": random.choice(FOODS),
        "term_medical": random.choice(MED_TERMS),
        "currency": random.choice(CURRENCY),
        "law": random.choice(LAW),
        "animal": random.choice(ANIMALS),
        "event": random.choice(EVENTS),
        "event_sports": random.choice(EVENT_SPORTS),
        "date": f"2026년 {month}월 {day}일",
        "date_duration": f"{random.randint(2, 18)}개월",
        "time_duration": f"{random.randint(10, 180)}분",
        "time": f"{hour:02d}:{minute:02d}",
        "quantity": str(random.randint(80, 12000)),
        "price": f"{random.randint(1, 80)}{random.choice(CURRENCY)}",
        "temperature": f"영하 {random.randint(1, 18)}도",
        "age": f"{random.randint(10, 70)}대",
    }


def generate_synthetic_sentences(count: int) -> Iterable[str]:
    for _ in range(count):
        template = random.choice(TEMPLATES)
        yield template.format(**make_context())


def deduplicate_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for text in items:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_jsonl", type=Path, default=Path("data/raw/news_10000.jsonl"))
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--real_news_path", type=Path, default=None)
    parser.add_argument("--real_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    target = max(1, args.num_samples)
    real_count = int(target * min(max(args.real_ratio, 0.0), 1.0))
    synthetic_count = target - real_count

    real_sentences: list[str] = []
    if args.real_news_path is not None:
        real_sentences = read_real_sentences(args.real_news_path)
    if len(real_sentences) < real_count:
        synthetic_count += real_count - len(real_sentences)
        real_count = len(real_sentences)

    sampled_real = random.sample(real_sentences, k=real_count) if real_count > 0 else []
    synthetic = list(generate_synthetic_sentences(synthetic_count * 2))

    merged = sampled_real + synthetic
    merged = deduplicate_keep_order(merged)

    if len(merged) < target:
        needed = target - len(merged)
        merged.extend(deduplicate_keep_order(generate_synthetic_sentences(needed * 3))[:needed])

    merged = merged[:target]
    rows = [{"id": i + 1, "text": text, "source": "real" if i < real_count else "synthetic"} for i, text in enumerate(merged)]
    write_jsonl(args.output_jsonl, rows)

    print(f"saved={len(rows)} path={args.output_jsonl}")


if __name__ == "__main__":
    main()
