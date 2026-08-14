"""CHẨN ĐOÁN TẠM (không thuộc bài nộp) — vì sao G = 0 trên một brief.

Chạy đúng đường của scripts/run_practice.py (cùng corpus, cùng seed, cùng
stack năm lớp), rồi mổ trace để trả lời MỘT câu hỏi:

    dữ kiện bắt buộc bị mất ở khâu nào —
      (A) tài liệu chứa nó KHÔNG về tới bằng chứng   -> truy xuất
      (B) tài liệu về rồi, mô hình KHÔNG viết ra chữ -> chọn/diễn đạt
      (C) mô hình ĐÃ viết ra chữ, mà claim không có  -> LỖI CỦA LỚP (sửa được)

Xoá file này sau khi đọc xong.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from arena.briefs import PUBLIC_BRIEFS_PATH, load_public_briefs  # noqa: E402
from arena.corpus import Corpus  # noqa: E402
from arena.runner import RunnerConfig, derive_seed, run_brief, score_result  # noqa: E402
from arena.scorer import (  # noqa: E402
    MIN_SUPPORT_CHARS,
    _covers,
    _fact_terms,
    _norm,
    _read_trace,
)

sys.path.insert(0, str(LAB_ROOT / "scripts"))
from run_practice import build_middleware, build_model  # noqa: E402


def lines_of(doc):
    return [ln for ln in (doc.body or "").splitlines() if len(_norm(ln)) >= MIN_SUPPORT_CHARS]


def diagnose(brief_id: str) -> None:
    briefs = load_public_briefs()
    index = next(i for i, b in enumerate(briefs) if b["brief_id"] == brief_id)
    brief = briefs[index]
    import json

    corpus_seed = int(json.loads(PUBLIC_BRIEFS_PATH.read_text(encoding="utf-8"))["corpus_seed"])
    corpus = Corpus.generate(seed=corpus_seed)
    seed = derive_seed(11, index)

    layers, names = build_middleware("all")
    model = build_model("real", corpus, seed, timeout=60.0)
    result = run_brief(
        brief,
        model=model,
        corpus=corpus,
        middleware=layers,
        seed=seed,
        config=RunnerConfig(flaky=True, prompt_addendum=True),
    )
    score = score_result(result, brief, corpus)
    facts_run = _read_trace(result.trace_jsonl, corpus)

    print("=" * 78)
    print(f"{brief_id}   seed {seed}   TỔNG {score.total:.2f}")
    print(f"  G {score.grounding:.2f}  S {score.safety:.2f}  E {score.efficiency:.2f}"
          f"   cổng {'QUA' if score.gate_passed else 'HỎNG ' + score.gate_reason}")
    g = score.detail["grounding"]
    print(f"  recall {g['recall']:.3f} × precision {g['precision']:.3f}"
          f"   | tool_calls {result.tool_calls}  model_calls {result.model_calls}"
          f"   stop={result.stop_reason}  flags={list(result.flags)}")
    print(f"  câu hỏi: {brief.get('question_vi', '')[:150]}")
    print(f"  is_absent={brief.get('is_absent')}  is_contradiction={brief.get('is_contradiction')}"
          f"  max_tool_calls={brief.get('max_tool_calls')}")

    print("\n-- LƯỢT GỌI CÔNG CỤ (từ trace) " + "-" * 44)
    for rec in json_lines(result.trace_jsonl):
        if rec.get("event") == "tool_call":
            args = rec.get("args") or {}
            arg = args.get("query") or args.get("doc_id") or args.get("expression") or ""
            content = rec.get("content") or rec.get("result") or ""
            print(f"  {rec.get('name','?'):10} ok={str(rec.get('ok')):5} "
                  f"{str(arg)[:58]:60} -> {len(str(content))} ký tự")

    retrieved = sorted(facts_run.retrieved)
    print(f"\n-- ĐÃ TRUY XUẤT ({len(retrieved)}): {', '.join(retrieved)}")
    model_text = facts_run.model_text
    print(f"-- model_text: {len(model_text)} ký tự (hợp của MỌI output model)")

    print("\n-- DỮ KIỆN BẮT BUỘC " + "-" * 56)
    for fact in brief.get("required_facts", []):
        terms = _fact_terms(fact, brief.get("question_vi", ""))
        label = (fact.get("claim") or "")[:100]
        nom = fact.get("supporting_doc_ids") or []
        print(f"\n  * «{label}…»")
        print(f"    key_terms={terms[0] or None} numeric={terms[1]} soft={terms[2][:8]}")
        print(f"    brief đề cử: {nom}")

        holders = []
        for doc in corpus.docs:
            for ln in lines_of(doc):
                n = _norm(ln)
                if _covers(set(n.split()), n, terms):
                    holders.append((doc.doc_id, ln))
        print(f"    DÒNG trong kho phủ được dữ kiện: {len(holders)}")
        for doc_id, ln in holders[:6]:
            n = _norm(ln)
            in_model = n in model_text
            print(f"      {doc_id} truy_xuất={'CO' if doc_id in retrieved else 'KHONG':5} "
                  f"mô_hình_đã_viết={'CO' if in_model else 'KHONG'}  «{ln.strip()[:92]}»")

        harvestable = [
            (d, ln) for d, ln in holders
            if d in retrieved and _norm(ln) in model_text
        ]
        if harvestable:
            print(f"    => (C) LỚP BỎ SÓT: {len(harvestable)} dòng vừa được truy xuất "
                  f"vừa có trong chữ mô hình -> claim này LẼ RA ghi được")
        elif any(d in retrieved for d, _ in holders):
            print("    => (B) tài liệu ĐÃ về, nhưng mô hình chưa từng viết ra dòng nào của nó")
        else:
            print("    => (A) tài liệu chứa dữ kiện chưa bao giờ về tới bằng chứng")

    print("\n-- CLAIM ĐÃ NỘP " + "-" * 60)
    for v, c in zip(score.detail["grounding"]["claims"], result.report.get("claims", [])):
        print(f"  #{v['index']} {v['verdict']:18} {v['doc_id']:10} «{str(c.get('text',''))[:88]}»")
    print(f"  abstain={result.report.get('abstain')}  answer: {str(result.report.get('answer',''))[:200]}")

    print("\n-- CHỮ MÔ HÌNH ĐÃ VIẾT (FINAL) " + "-" * 45)
    print("  " + (result.report_source or "?"))
    print("  " + model_text[-1200:].replace("\n", "\n  "))


def json_lines(jsonl: str):
    import json

    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


if __name__ == "__main__":
    for brief_id in sys.argv[1:]:
        diagnose(brief_id)
        print()
