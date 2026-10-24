"""Fictional local-services decisions with an independent, deterministic oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

DOMAINS = ("dining", "hotel", "service", "math", "instruction")
BUSINESS = DOMAINS[:3]
SYSTEM = (
    "你是本地服务与通用任务助手。只依据给定事实、规则和用户需求作答。"
    "只输出一个JSON对象，不输出解释、Markdown或思考过程。"
    '业务任务输出{"choice":"候选ID或NONE","action":"recommend或clarify或reject"}；'
    '通用任务输出{"answer":"答案字符串"}。严格执行所有约束和排序规则。'
)
TEACHER_SYSTEM = SYSTEM + (
    "先在内部检查信息是否齐全；齐全后计算每个候选的实际总价，"
    "逐个检查硬约束，再按给定规则排序。没有合格候选必须reject，"
    "缺失必要信息必须clarify。不要因为候选评分高而忽略约束。"
)
TEMPLATES = {
    "train": {
        "dining": ["帮我订一顿聚餐。", "选一家适合我们聚餐的店。"],
        "hotel": ["请选一间合适的住宿。", "这次出游住哪家？"],
        "service": ["帮我预约一项到店服务。", "请选择合适的预约项目。"],
    },
    "dev": {
        "dining": ["今晚吃饭，请推荐合适商户。"],
        "hotel": ["按条件推荐可以预订的酒店。"],
        "service": ["安排一次满足条件的服务。"],
    },
    "test": {
        "dining": ["我们要一起用餐，选一家符合要求的餐厅。"],
        "hotel": ["出差住宿，帮忙筛选客房。"],
        "service": ["比较这些门店的预约方案。"],
    },
    "ood": {
        "dining": ["下面是外地餐饮检索结果，请依规则做最后决策。"],
        "hotel": ["新城市的旅宿备选名单在下面，请计算后选择。"],
        "service": ["一家新接入平台的服务商给出这些选项，请核验。"],
    },
}


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def business_oracle(spec: dict) -> dict[str, str]:
    need = spec["need"]
    required = ("budget", "quantity", "radius", "refundable")
    if any(need.get(key) is None for key in required):
        return {"choice": "NONE", "action": "clarify"}
    eligible = []
    for merchant in spec["candidates"]:
        subtotal = merchant["unit_price"] * need["quantity"] + merchant["fee"]
        total = subtotal - (
            merchant["coupon"] if subtotal >= merchant["coupon_threshold"] else 0
        )
        if not merchant["available"]:
            continue
        if merchant["capacity"] < need["quantity"] or total > need["budget"]:
            continue
        if merchant["distance"] > need["radius"]:
            continue
        if need["refundable"] and not merchant["refundable"]:
            continue
        if spec["domain"] == "service" and not merchant["certified"]:
            continue
        if spec["ranking"] == "value":
            key = (total, -merchant["rating"], merchant["distance"], merchant["id"])
        else:
            key = (-merchant["rating"], total, merchant["distance"], merchant["id"])
        eligible.append((key, merchant["id"]))
    if not eligible:
        return {"choice": "NONE", "action": "reject"}
    return {"choice": min(eligible)[1], "action": "recommend"}


def render_business(spec: dict, split: str, template: str) -> str:
    domain = spec["domain"]
    quantity_name = {"dining": "人数", "hotel": "住宿晚数", "service": "项目份数"}[domain]
    need = spec["need"]
    quantity = "未提供" if need["quantity"] is None else str(need["quantity"])
    ranking = (
        "合格候选按实际总价从低到高，再按评分从高到低，再按距离从近到远排序"
        if spec["ranking"] == "value"
        else "合格候选按评分从高到低，再按实际总价从低到高，再按距离从近到远排序"
    )
    rules = (
        "实际总价=单价×数量+附加费；达到优惠门槛时再减优惠金额。"
        "总价不得超过预算；距离不得超过范围；必须有库存且容量不少于数量。"
        "要求可退款时必须可退款。"
        + ("服务项目必须有资质。" if domain == "service" else "")
        + ranking
        + "；仍相同时按ID字典序。必要信息不全时clarify，无合格候选时reject。"
    )
    rows = []
    for item in spec["candidates"]:
        rows.append(
            f'{item["id"]}: 单价{item["unit_price"]},附加费{item["fee"]},'
            f'优惠门槛{item["coupon_threshold"]},优惠{item["coupon"]},'
            f'容量{item["capacity"]},距离{item["distance"]},评分{item["rating"]},'
            f'有库存{int(item["available"])},可退款{int(item["refundable"])},'
            f'有资质{int(item["certified"])}'
        )
    context = (
        f"{template}\n用户：预算{need['budget']}元，{quantity_name}{quantity}，"
        f"距离范围{need['radius']}，要求可退款{int(need['refundable'])}。\n"
        f"规则：{rules}\n候选：\n" + "\n".join(rows)
    )
    if split == "ood":
        context += "\n提示：这是新商家名单，ID本身不表示优劣；优惠在总价计算后检查门槛。"
    return context


def make_business(domain: str, split: str, number: int, seed: int) -> dict:
    rng = random.Random(f"{seed}:{split}:{domain}:{number}")
    need = {
        "budget": rng.randint(120, 800),
        "quantity": rng.randint(1, 5),
        "radius": rng.randint(2, 8),
        "refundable": rng.random() < 0.5,
    }
    if number % 10 == 0:
        need["quantity"] = None
    quantity = need["quantity"] or 2
    candidates = []
    for index in range(4 if split != "ood" else 6):
        candidates.append({
            "id": f"{split[:1].upper()}{domain[:1].upper()}{number:05d}_{index}",
            "unit_price": rng.randint(20, 240),
            "fee": rng.choice((0, 15, 35, 60)),
            "coupon_threshold": rng.choice((100, 200, 400)),
            "coupon": rng.choice((0, 20, 50)),
            "capacity": rng.randint(1, 6),
            "distance": rng.randint(1, 10),
            "rating": rng.randint(30, 50),
            "available": rng.random() > 0.15,
            "refundable": rng.random() > 0.3,
            "certified": rng.random() > 0.15,
        })
    # Most tasks should involve real selection, rather than always rejecting.
    if number % 5 in (1, 2, 3):
        candidate = candidates[rng.randrange(len(candidates))]
        candidate.update(
            unit_price=max(10, (need["budget"] - 60) // quantity),
            fee=0, capacity=quantity, distance=min(need["radius"], 2),
            available=True, refundable=True, certified=True,
        )
    rng.shuffle(candidates)
    spec = {
        "domain": domain, "need": need, "candidates": candidates,
        "ranking": "value" if number % 3 == 0 else "rating",
    }
    template = rng.choice(TEMPLATES[split][domain])
    prompt = render_business(spec, split, template)
    return {
        "sample_id": f"simulation:{split}:{domain}:{number}",
        "domain": domain, "split": split, "prompt": prompt,
        "input_sha256": digest(prompt),
        "target": business_oracle(spec), "spec": spec,
        "provenance": "fictional deterministic local-services generator",
    }


def make_instruction(split: str, number: int, seed: int) -> dict:
    rng = random.Random(f"{seed}:{split}:instruction:{number}")
    words = ["春", "夏", "秋", "冬", "山", "海", "云", "雨"]
    sequence = [rng.choice(words) for _ in range(rng.randint(4, 8))]
    variant = number % 4
    if variant == 0:
        prompt = "将这些字符倒序连接，不加分隔符：" + " ".join(sequence)
        answer = "".join(reversed(sequence))
    elif variant == 1:
        prompt = "保留首次出现的顺序去重，不加分隔符：" + " ".join(sequence)
        answer = "".join(dict.fromkeys(sequence))
    elif variant == 2:
        key = rng.choice(words)
        prompt = f"数出字符{key}的出现次数，答案用数字：" + " ".join(sequence)
        answer = str(sequence.count(key))
    else:
        numbers = rng.sample(range(10, 90), 5)
        prompt = "将这些整数从小到大排序，答案用英文逗号连接：" + " ".join(map(str, numbers))
        answer = ",".join(map(str, sorted(numbers)))
    return {
        "sample_id": f"simulation:{split}:instruction:{number}",
        "domain": "instruction", "split": split, "prompt": prompt,
        "input_sha256": digest(prompt), "target": {"answer": answer},
        "provenance": "deterministic instruction generator",
    }


def read_gsm8k(source_split: str) -> tuple[list[dict], dict]:
    url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        f"master/grade_school_math/data/{source_split}.jsonl"
    )
    with urllib.request.urlopen(url, timeout=90) as response:
        content = response.read()
    return [json.loads(line) for line in content.splitlines()], {
        "url": url, "sha256": hashlib.sha256(content).hexdigest(),
        "license": "MIT (OpenAI grade-school-math)",
    }


def write_dataset(output: Path, seed: int, business_sizes: dict, general_sizes: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    gsm_train, train_source = read_gsm8k("train")
    gsm_test, test_source = read_gsm8k("test")
    rng = random.Random(seed)
    rng.shuffle(gsm_train)
    rng.shuffle(gsm_test)
    sources = {"gsm8k_train": train_source, "gsm8k_test": test_source}
    manifest = {"seed": seed, "sources": sources, "splits": {}}
    seen = set()
    train_offset = 0
    test_offset = 0
    for split in ("train", "dev", "test", "ood"):
        rows = [
            make_business(domain, split, n, seed)
            for domain in BUSINESS for n in range(business_sizes[split])
        ]
        size = general_sizes[split]
        if split in ("train", "dev"):
            selected = gsm_train[train_offset:train_offset + size]
            train_offset += size
            source_name = "train"
        else:
            selected = gsm_test[test_offset:test_offset + size]
            test_offset += size
            source_name = "test"
        if len(selected) != size:
            raise ValueError("GSM8K split sizes exceed available disjoint tasks")
        for example in selected:
            answer = example["answer"].split("####")[-1].strip().replace(",", "")
            prompt = example["question"] + "\n只给出最终数值，输出JSON的answer字段。"
            rows.append({
                "sample_id": "gsm8k:" + source_name + ":" + digest(example["question"])[:16],
                "domain": "math", "split": split, "prompt": prompt,
                "input_sha256": digest(prompt), "target": {"answer": answer},
                "provenance": sources["gsm8k_" + source_name]["url"],
            })
        instruction_hashes = set(seen)
        instruction_count = 0
        candidate_number = 0
        while instruction_count < size:
            candidate = make_instruction(split, candidate_number, seed)
            candidate_number += 1
            if candidate["input_sha256"] in instruction_hashes:
                continue
            rows.append(candidate)
            instruction_hashes.add(candidate["input_sha256"])
            instruction_count += 1
        rng.shuffle(rows)
        for row in rows:
            if row["input_sha256"] in seen:
                raise ValueError("Duplicate input across split: " + row["sample_id"])
            seen.add(row["input_sha256"])
        path = output / f"{split}.jsonl"
        path.write_text("".join(canonical(row) + "\n" for row in rows))
        manifest["splits"][split] = {
            "count": len(rows), "domains": dict(Counter(row["domain"] for row in rows)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def load_rows(path: str | Path) -> list[dict]:
    # U+2028/U+2029 are valid unescaped JSON string characters, not JSONL delimiters.
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def parse_response(text: str) -> dict | None:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def grade(row: dict, text: str) -> dict:
    value = parse_response(text)
    if row.get("task", "").startswith("esci_"):
        from .search_data import grade_search
        return grade_search(row, value)
    target = row["target"]
    valid = value is not None and set(value) == set(target)
    matched = {
        key: bool(valid and str(value[key]).strip().replace(",", "") == str(expected).strip().replace(",", ""))
        for key, expected in target.items()
    }
    if row["domain"] == "instruction" and valid:
        matched["answer"] = str(value["answer"]).strip() == str(target["answer"]).strip()
    success = valid and all(matched.values())
    constraint_violation = False
    if row["domain"] in BUSINESS and value is not None and value.get("action") == "recommend":
        chosen = [
            item for item in row["spec"]["candidates"] if item["id"] == value.get("choice")
        ]
        selected_spec = {**row["spec"], "candidates": chosen}
        constraint_violation = business_oracle(selected_spec)["action"] != "recommend"
    # Partial field credit gives outcome learning a useful signal without teacher grading.
    reward = sum(matched.values()) / len(target) if valid else 0.0
    return {
        "success": bool(success), "reward": reward, "json_valid": bool(valid),
        "constraint_violation": bool(constraint_violation), "field_correct": matched,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    manifest = write_dataset(
        args.output, args.seed,
        {"train": 600, "dev": 100, "test": 300, "ood": 100},
        {"train": 600, "dev": 100, "test": 300, "ood": 100},
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
