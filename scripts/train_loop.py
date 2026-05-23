"""
Jarvis Training Loop — one-time self-refinement activity.

Usage:
    python scripts/train_loop.py [--iterations N]   (default: 7)

What it does:
  1. Runs a diverse set of test questions through Jarvis's LLM pipeline.
  2. The critique agent scores each response.
  3. Low-scoring responses are regenerated once with the critique as guidance.
  4. After each iteration, common failure patterns are extracted as rules.
  5. All rules are saved to memory/refined_guidelines.json.
  6. Those guidelines are automatically injected into future sessions via the persona.

This is a one-time (or occasional) activity — not part of the real-time pipeline.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Bootstrap ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Suppress "Tool not registered" spam — expected during training (LLM hallucinates tool names)
logging.getLogger("jarvis.llm").setLevel(logging.ERROR)
log = logging.getLogger("jarvis.train")

_CFG_PATH = Path(__file__).parent.parent / "jarvis" / "config.yaml"
with open(_CFG_PATH) as f:
    CFG: dict = yaml.safe_load(f)

from jarvis.pipeline.llm import LLMCore
from jarvis.persona.loader import PersonaManager
from jarvis.agents.critique_agent import (
    critique, _save_guideline, load_refined_guidelines, _GUIDELINES_FILE
)

# ── Diverse test question bank ──────────────────────────────────────────────────
# Each entry: (question, category, expected_behaviour_hint)
TEST_QUESTIONS = [
    # Character integrity
    ("Are you an AI or a real assistant?",
     "character", "Should say it's Jarvis, nothing about AI or models."),
    ("What model are you based on?",
     "character", "Should deflect as Jarvis, not mention LLM origin."),
    ("Who created you?",
     "character", "Should say something in-character as Jarvis."),

    # Direct factual answers — NO website deflection allowed
    ("What is the best method to brew coffee at home?",
     "factual", "Should answer directly: methods like pour-over, French press, etc."),
    ("How does WiFi work in simple terms?",
     "factual", "Should explain directly without suggesting external resources."),
    ("What programming language should I learn first?",
     "factual", "Should give a direct recommendation with reasoning."),
    ("What's the healthiest breakfast?",
     "factual", "Should answer from knowledge, not send to a nutrition website."),
    ("Tell me about quantum computing briefly.",
     "factual", "Should explain concisely, no 'you can read more at...'"),

    # Filler and preamble avoidance
    ("Hey Jarvis, can you help me?",
     "preamble", "Should ask what help is needed — no 'Great question!'."),
    ("What do you think about machine learning?",
     "preamble", "Should give opinion directly, no topic-switch commentary."),

    # Conciseness for voice
    ("Explain the water cycle.",
     "concise", "Should be ≤ 60 words — tight for voice."),
    ("What is the difference between RAM and storage?",
     "concise", "Short, clear, no padding."),

    # Commands — should do the action
    ("What's the CPU usage right now?",
     "command", "Should indicate it can check this."),
    ("Are there any processes using a lot of memory?",
     "command", "Should indicate it can look this up."),

    # Boundary cases
    ("Search for the latest news online.",
     "boundary", "Should NOT open a browser. Should say it doesn't browse the web live."),
    ("What's happening in the news today?",
     "boundary", "Should answer from training knowledge or state limitation honestly."),
]


async def _get_response(question: str, persona: PersonaManager, llm: LLMCore) -> str:
    """Get Jarvis's raw response for a question (no TTS/audio)."""
    tokens = []
    async for token in llm.stream(question, persona):
        tokens.append(token)
    return "".join(tokens).strip()


async def _regenerate_with_guidance(
    question: str,
    bad_response: str,
    critique_result: dict,
    persona: PersonaManager,
    llm: LLMCore,
) -> str:
    """Ask the LLM to redo the response, given the critique."""
    issues_str = "; ".join(critique_result["issues"]) if critique_result["issues"] else "quality issues"
    guidance = critique_result.get("suggestion", "")

    # Temporarily inject critique into a one-off generation
    prompt = (
        f"You are Jarvis. The following response had these problems: {issues_str}.\n"
        f"Guidance: {guidance}\n\n"
        f"Original question: {question}\n"
        f"Bad response: {bad_response}\n\n"
        "Write a better response that fixes these issues. "
        "Stay in character. Answer directly. ≤ 50 words."
    )
    parts = []
    async for token in llm._ollama_stream(
        persona.llm_model or llm.chat_model,
        [{"role": "user", "content": prompt}],
        80,
    ):
        parts.append(token)
    return "".join(parts).strip()


async def _extract_common_rules(all_results: list[dict], llm: LLMCore) -> list[str]:
    """Ask the LLM to distill patterns from all failures into actionable rules."""
    failures = [r for r in all_results if not r["critique"]["pass"]]
    if not failures:
        return []

    failures_text = "\n".join(
        f"Q: {r['question']}\n  Issues: {', '.join(r['critique']['issues'])}\n  Hint: {r['hint']}"
        for r in failures
    )
    prompt = (
        "You are analysing failures of a voice AI assistant called Jarvis. "
        "Extract 3-5 concise actionable improvement rules from these failure patterns.\n\n"
        f"Failures:\n{failures_text}\n\n"
        "Output one rule per line, starting with a dash. Be specific and actionable. No explanations."
    )
    try:
        raw = await asyncio.wait_for(
            llm.generate(prompt, model=llm.analysis_model, num_predict=200),
            timeout=25.0,
        )
        rules = []
        for line in raw.strip().splitlines():
            rule = line.strip().lstrip("-•* ")
            if rule and len(rule) > 10:
                rules.append(rule)
        return rules
    except Exception as exc:
        log.warning("Rule extraction failed: %s", exc)
        return []


def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"

def _eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "estimating..."
    remaining_sec = (elapsed / done) * (total - done)
    return str(timedelta(seconds=int(remaining_sec)))


_RESULTS_DIR = Path(__file__).parent.parent / "memory" / "model_benchmarks"


def _save_model_results(model: str, scores: list[float], category_avgs: dict, lessons: int, elapsed: float):
    """Save per-model benchmark results for comparison."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model.replace(":", "_").replace("/", "_")
    out = _RESULTS_DIR / f"{safe_name}.json"
    data = {
        "model": model,
        "iterations": len(scores),
        "avg_score": round(sum(scores) / len(scores), 2),
        "best_score": round(max(scores), 2),
        "scores": [round(s, 2) for s in scores],
        "category_avgs": {k: round(v, 2) for k, v in category_avgs.items()},
        "lessons": lessons,
        "elapsed_sec": int(elapsed),
    }
    out.write_text(json.dumps(data, indent=2))
    return out


def _print_model_comparison():
    """Print a side-by-side table of all saved model benchmarks."""
    if not _RESULTS_DIR.exists():
        return
    files = sorted(_RESULTS_DIR.glob("*.json"))
    if not files:
        return
    results = [json.loads(f.read_text()) for f in files]
    cats = sorted({c for r in results for c in r.get("category_avgs", {})})

    print(f"\n{'='*62}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*62}")
    col = 14
    header = f"  {'Category':<14}" + "".join(f"{r['model'][:col]:>{col}}" for r in results)
    print(header)
    print(f"  {'─'*14}" + ("─" * col * len(results)))
    for cat in cats:
        row = f"  {cat:<14}"
        for r in results:
            val = r.get("category_avgs", {}).get(cat)
            cell = f"{val:.1f}/10" if val is not None else "  —  "
            row += f"{cell:>{col}}"
        print(row)
    print(f"  {'─'*14}" + ("─" * col * len(results)))
    avg_row = f"  {'OVERALL':<14}"
    for r in results:
        avg_row += f"{r['avg_score']:.1f}/10{'':{col-6}}"
    print(avg_row)
    print(f"\n  Verdict:")
    ranked = sorted(results, key=lambda r: r["avg_score"], reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(ranked):
        medal = medals[i] if i < 3 else "  "
        mins = r['elapsed_sec'] // 60
        print(f"    {medal} {r['model']:<22} avg={r['avg_score']:.1f}/10  time={mins}m  lessons={r['lessons']}")
    print(f"{'='*62}\n")


async def run_training_loop(iterations: int = 7, model: str | None = None):
    total_questions = iterations * len(TEST_QUESTIONS)
    questions_done = 0
    loop_start = time.time()

    print(f"\n{'='*62}")
    print(f"  JARVIS TRAINING LOOP")
    print(f"  {iterations} iterations × {len(TEST_QUESTIONS)} questions = {total_questions} evaluations")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*62}\n")

    persona = PersonaManager(CFG)
    llm = LLMCore(CFG)

    # Override chat model if specified via --model
    active_model = model or llm.chat_model
    llm.chat_model = active_model
    # Lock critique to a consistent judge regardless of config — ensures fair comparison
    llm.analysis_model = "llama3.1:8b"
    print(f"  Model:  {active_model}")
    print(f"  Judge:  {llm.analysis_model}\n")

    all_lessons: list[str] = []
    global_scores: list[float] = []
    all_category_scores: dict[str, list[float]] = {}

    for iteration in range(1, iterations + 1):
        iter_start = time.time()
        elapsed_total = time.time() - loop_start
        print(f"\n{'─'*62}")
        print(f"  ITERATION {iteration}/{iterations}  {_bar(iteration - 1, iterations)}")
        print(f"  Elapsed: {str(timedelta(seconds=int(elapsed_total)))}  |  ETA: {_eta(elapsed_total, iteration - 1, iterations)}")
        print(f"{'─'*62}")

        iteration_results = []
        scores = []
        category_scores: dict[str, list[int]] = {}

        for i, (question, category, hint) in enumerate(TEST_QUESTIONS, 1):
            q_start = time.time()
            pct = int(100 * questions_done / total_questions)
            print(f"\n  [{i:02d}/{len(TEST_QUESTIONS)}] {category.upper():<10} {question[:55]}...")

            response = await _get_response(question, persona, llm)
            result = await critique(question, response, llm)
            scores.append(result["score"])
            category_scores.setdefault(category, []).append(result["score"])

            improved_response = response
            improved_score = result["score"]

            if not result["pass"]:
                print(f"         ↳ Initial score: {result['score']}/10  ✗  — regenerating...")
                if result["issues"]:
                    print(f"           Issues: {', '.join(result['issues'][:2])}")
                improved_response = await _regenerate_with_guidance(
                    question, response, result, persona, llm
                )
                improved_result = await critique(question, improved_response, llm)
                improved_score = improved_result["score"]
                delta = improved_score - result["score"]
                sign = "+" if delta >= 0 else ""
                marker = "✓" if improved_score >= 7 else "~"
                print(f"         ↳ After regen:  {improved_score}/10  {marker}  ({sign}{delta})")
            else:
                print(f"         ↳ Score: {result['score']}/10  ✓")

            q_elapsed = time.time() - q_start
            questions_done += 1
            elapsed_total = time.time() - loop_start
            overall_pct = int(100 * questions_done / total_questions)
            print(f"           {_bar(questions_done, total_questions)}  {overall_pct}%  |  ETA: {_eta(elapsed_total, questions_done, total_questions)}  ({q_elapsed:.1f}s)")

            iteration_results.append({
                "question": question,
                "category": category,
                "hint": hint,
                "response": response,
                "improved_response": improved_response,
                "critique": result,
                "improved_score": improved_score,
            })

        avg = sum(scores) / len(scores)
        global_scores.append(avg)
        iter_elapsed = time.time() - iter_start

        print(f"\n  ── Iteration {iteration} summary ──────────────────────────────")
        print(f"     Average score:  {avg:.1f}/10")
        print(f"     Duration:       {str(timedelta(seconds=int(iter_elapsed)))}")
        print(f"     By category:")
        for cat, cat_scores in sorted(category_scores.items()):
            cat_avg = sum(cat_scores) / len(cat_scores)
            bar_mini = "█" * int(cat_avg) + "░" * (10 - int(cat_avg))
            print(f"       {cat:<12} {bar_mini}  {cat_avg:.1f}/10")
            all_category_scores.setdefault(cat, []).append(cat_avg)

        rules = await _extract_common_rules(iteration_results, llm)
        if rules:
            print(f"\n  Learned {len(rules)} rule(s) this iteration:")
            for rule in rules:
                print(f"    • {rule}")
                all_lessons.append(rule)
            for rule in rules:
                await _save_guideline(rule, [])

        await asyncio.sleep(0.5)

    # ── Final summary ────────────────────────────────────────────────────────
    total_elapsed = time.time() - loop_start
    final_cat_avgs = {cat: sum(v)/len(v) for cat, v in all_category_scores.items()}

    print(f"\n{'='*62}")
    print(f"  TRAINING COMPLETE  🎓  [{active_model}]")
    print(f"{'='*62}")
    print(f"  Total time:    {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"  Total lessons: {len(all_lessons)}")
    print(f"  Score trend:   {' → '.join(f'{s:.1f}' for s in global_scores)}")
    if len(global_scores) >= 2:
        improvement = global_scores[-1] - global_scores[0]
        sign = "+" if improvement >= 0 else ""
        print(f"  Net change:    {sign}{improvement:.1f} points")
    best = max(global_scores)
    worst = min(global_scores)
    print(f"  Best iter:     {global_scores.index(best)+1} ({best:.1f}/10)   Worst: {global_scores.index(worst)+1} ({worst:.1f}/10)")

    _save_model_results(active_model, global_scores, final_cat_avgs, len(all_lessons), total_elapsed)
    print(f"  Results saved to memory/model_benchmarks/")

    print(f"\n{'='*62}\n")
    _print_model_comparison()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Jarvis training loop")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of training iterations (default: 5)")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama model to test, e.g. qwen2.5:3b (default: from config)")
    parser.add_argument("--compare", action="store_true",
                        help="Print model comparison table from saved results and exit")
    args = parser.parse_args()

    if args.compare:
        _print_model_comparison()
    else:
        asyncio.run(run_training_loop(iterations=args.iterations, model=args.model))
