"""
Benchmark: Nemotron alternatives vs. the 550B baseline (AURA v0.2).

READ-ONLY w.r.t. AURA: this does NOT touch settings.py, .env, nemotron.py, or
factory.py. It builds its own OpenAI client and sets ``model=`` per call, so the
production default (nvidia/nemotron-3-ultra-550b-a55b, the baseline row below)
is left exactly as-is.  The 550B model is the baseline reference; the smaller
models are the "alternatives".

Metrics (single streaming call per model x prompt, max_tokens=48, temp=0.2):
  ok/error, ttft_ms, total_ms, prompt_tokens, completion_tokens, tokens/sec,
  output_words, accuracy (expected-answer substring proxy), conciseness
  (output in 20..50 words, per AURA's voice rules), markdown-free rate.

Robust to the free endpoint's transient "ResourceExhausted (16/16)": retries
with exponential backoff, never crashes on a single failure.
Requires AURA_NVIDIA_API_KEY to be set in .env (not printed here).
"""

from __future__ import annotations

from time import perf_counter, sleep

from openai import APIError, OpenAI

from aura.brain.nemotron import SYSTEM_PROMPT  # read-only import for parity
from aura.config.settings import settings

# --------------------------------------------------------------------------- #
# Shared prompt set: short factual QA mirroring AURA's "factual question" use.
# Each tuple is (prompt, expected_substring_for_accuracy_proxy).
# --------------------------------------------------------------------------- #
PROMPTS: list[tuple[str, str]] = [
    ("What's 15 percent of 80?", "12"),
    ("What is the capital of France?", "Paris"),
    ("How many bones are in an adult human body?", "206"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Who wrote the play 'Romeo and Juliet'?", "Shakespeare"),
    ("What is the largest planet in our solar system?", "Jupiter"),
]

# 550B = baseline. Alternatives span size tiers available on the free endpoint.
MODELS: list[tuple[str, str]] = [
    ("baseline", "nvidia/nemotron-3-ultra-550b-a55b"),
    ("alt",      "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
    ("alt",      "nvidia/nemotron-3.5-lightning-30b-a3b"),
    ("alt",      "nvidia/nemotron-3-super-120b-a12b"),
    ("alt",      "nvidia/nemotron-4-340b-instruct"),
]

MAX_TOKENS = 48
TEMPERATURE = 0.2  # mirrors AURA's production setting for apples-to-apples

CLIENT: OpenAI | None = None

def _is_resource_exhausted(exc: BaseException) -> bool:
    return isinstance(exc, APIError) and "resourceexhausted" in str(exc).lower()


def call(model: str, prompt: str) -> dict:
    """One streaming request; returns a result dict (never raises)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    started = perf_counter()
    ttft_ms: float | None = None
    parts: list[str] = []
    usage = None
    for attempt in range(4):  # 1 + 3 retries for transient 16/16
        try:
            stream = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                delta = None
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta.content
                if ttft_ms is None and delta is not None:
                    ttft_ms = (perf_counter() - started) * 1000.0
                if delta:
                    parts.append(delta)
                u = getattr(chunk, "usage", None)
                if u is not None:
                    usage = u
            text = "".join(parts)
            total_ms = (perf_counter() - started) * 1000.0
            if not text or not text.strip():
                return {
                    "ok": False, "error": "empty_reply", "text": "",
                    "ttft_ms": round(ttft_ms or total_ms, 1),
                    "total_ms": round(total_ms, 1), "in": 0, "out": 0,
                }
            in_tok = usage.prompt_tokens if usage else len(prompt) // 4
            out_tok = usage.completion_tokens if usage else len(text) // 4
            return {
                "ok": True,
                "text": text.strip(),
                "ttft_ms": round(ttft_ms or total_ms, 1),
                "total_ms": round(total_ms, 1),
                "in": in_tok,
                "out": out_tok,
            }
        except APIError as exc:
            if _is_resource_exhausted(exc) and attempt < 3:
                delay = 0.5 * (2 ** attempt)
                print(f"  [16/16 retry {attempt+1}/3 in {delay}s for {model}]", flush=True)
                sleep(delay)
                continue
            return {
                "ok": False, "error": str(exc).splitlines()[0][:160],
                "text": "", "ttft_ms": 0.0, "total_ms": 0.0, "in": 0, "out": 0,
            }
        except Exception as exc:  # noqa: BLE001 - benchmark keeps going
            return {
                "ok": False, "error": type(exc).__name__ + ": " + str(exc)[:160],
                "text": "", "ttft_ms": 0.0, "total_ms": 0.0, "in": 0, "out": 0,
            }
    return {
        "ok": False, "error": "resource_exhausted_retries_exhausted",
        "text": "", "ttft_ms": 0.0, "total_ms": 0.0, "in": 0, "out": 0,
    }


def _is_markdown_free(text: str) -> bool:
    no_block = "```" not in text
    no_list = ("\n#" not in text) and ("\n-" not in text) and ("\n*" not in text)
    return no_block and no_list

def main() -> None:
    global CLIENT
    print("AURA v0.2 - Nemotron alternatives vs 550B baseline benchmark")
    print("=" * 78)
    print(f"provider base_url : {settings.nvidia_base_url}")
    print(f"key present       : {bool(settings.nvidia_api_key)}")
    print(f"production model  : {settings.model}  (unchanged; 550B = baseline row)")
    print(f"temperature       : {TEMPERATURE}  max_tokens: {MAX_TOKENS}")
    print(f"prompts           : {len(PROMPTS)}   models: {len(MODELS)}")
    print("=" * 78)
    CLIENT = OpenAI(
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
        timeout=60.0,
    )

    roll: dict[str, dict] = {}
    for _label, model in MODELS:
        roll[model] = {
            "n": 0, "ok": 0, "err": 0,
            "ttft": [], "total": [], "in": [], "out": [], "words": [],
            "correct": 0, "concise": 0, "md_free": 0,
        }

    for idx, (prompt, expected) in enumerate(PROMPTS, 1):
        print(f"\n[{idx}/{len(PROMPTS)}] {prompt}", flush=True)
        for _label, model in MODELS:
            r = call(model, prompt)
            row = roll[model]
            row["n"] += 1
            if r["ok"]:
                row["ok"] += 1
                row["ttft"].append(r["ttft_ms"])
                row["total"].append(r["total_ms"])
                row["in"].append(r["in"])
                row["out"].append(r["out"])
                words = len(r["text"].split())
                row["words"].append(words)
                if expected.lower() in r["text"].lower():
                    row["correct"] += 1
                if 20 <= words <= 50:
                    row["concise"] += 1
                if _is_markdown_free(r["text"]):
                    row["md_free"] += 1
                print(
                    f"  {model:55s} ok=1 ttft={r['ttft_ms']:.0f}ms "
                    f"total={r['total_ms']:.0f}ms in={r['in']} out={r['out']} "
                    f"words={words} :: {r['text'][:60]}", flush=True
                )
            else:
                row["err"] += 1
                print(f"  {model:55s} FAIL {r['error'][:80]}", flush=True)

    print("\n" + "=" * 78)
    print("SUMMARY (550B = baseline)")
    print("=" * 78)
    hdr = (
        f"{'model':55s} {'ok':>3} {'ttft':>6} {'tot':>6} {'in':>5} {'out':>5} "
        f"{'wds':>5} {'acc':>4} {'con':>4} {'md':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for model, row in roll.items():
        n = row["n"]
        ok = row["ok"]
        acc = f"{100*row['correct']/n:.0f}%" if ok else " -"
        con = f"{100*row['concise']/n:.0f}%" if ok else " -"
        md = f"{100*row['md_free']/n:.0f}%" if ok else " -"
        ttft = f"{sum(row['ttft'])/len(row['ttft']):.0f}" if row["ttft"] else "  -"
        total = f"{sum(row['total'])/len(row['total']):.0f}" if row["total"] else "  -"
        in_t = f"{sum(row['in'])/len(row['in']):.0f}" if row["in"] else "  -"
        out_t = f"{sum(row['out'])/len(row['out']):.0f}" if row["out"] else "  -"
        wds = f"{sum(row['words'])/len(row['words']):.0f}" if row["words"] else "  -"
        print(
            f"{model:55s} {ok:3d} {ttft:>6} {total:>6} {in_t:>5} {out_t:>5} "
            f"{wds:>5} {acc:>4} {con:>4} {md:>4}"
        )
    print("=" * 78)
    print("Notes: acc = expected-answer substring proxy; con = output in 20..50 words;")
    print("md = no markdown/bullets/code-blocks. ms & token counts are averages.")
    print("Costs are NOT shown; token usage is reported for relative comparison.")


if __name__ == "__main__":
    main()
