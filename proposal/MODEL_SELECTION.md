# Model Selection

## Recommendation

**`anthropic/claude-haiku-5.5`, reasoning off, for players and referee.**

- 0% player parse failures and 0% IAR over 120 decisions (targets: under 2% and under 5%).
- Strongest coordination of all finalists: most directives sent (213), highest MAAR and CCAR counts, and by far the highest use of team memory (MAUR 87%).
- High memory use gives the memory ablation (A vs D) a clear signal to measure.
- About $0.084 to $0.108 per 20 events, so about **$1.20 to $1.50 per 280-event game** and about **$71 to $91 for 60 games**.

Mistral Small 3.2 is the runner-up. It is valid and uses memory (MAUR 18%) at about a quarter of the cost (about $24 for 60 games). Qwen and DeepSeek v4.1 are just as valid, but almost never reference team memory (0/120 and 1/120). That would make MAUR 0 by construction and leave the memory ablation with nothing to measure.

## Results

Same seed (101) for every run. Each model played as all 6 players and as the referee. Raw logs are in `runs/model_selection/` (not committed).

| Model | Reasoning | Events | Player parse fail | IAR | Referee parse fail | Out tokens per player call | Cost |
|---|---|---|---|---|---|---|---|
| mistralai/mistral-small-3.2-24b-instruct | none | 20 | 0% | 0% | 3/20 (see notes) | 137 | $0.028 |
| deepseek/deepseek-v4.1-flash | off | 20 | 0% | 0% | 0/20 | 154 | $0.026 |
| qwen/qwen3.7-flash | off | 20 | 0% | 0% | 1/20 | 159 | $0.013 |
| anthropic/claude-haiku-5.5 | off | 20 | 0% | 0% | 2/20 | 424 | $0.084 to $0.108 |
| mistralai/mistral-small-3.2-24b-instruct | none | 5 | 0% | 0% | 0/5 | 133 | $0.007 to $0.008 |
| qwen/qwen3.7-flash | off | 5 | 0% | 0% | 0/5 | 149 | $0.003 to $0.004 |
| openai/gpt-oss-120b | low | 5 | 0% | 0% | 1/5 | 296 | $0.004 to $0.006 |
| openai/gpt-4.1-nano | none | 5 | 0% | 6.7% | 1/5 | 161 | $0.005 to $0.009 |
| deepseek/deepseek-v4-flash | off | 5 | 0% | 10% | 0/5 | 162 | $0.006 |
| google/gemini-2.5-flash-lite | off | 0 | not run: upstream rate limit (429) | | | | $0 |


## Coordination (20-event finalists)

Computed with `compute_game_metrics` from the same logs. Numerator / denominator summed over both teams.

| Model | Directives sent | MAAR | CCAR | MAUR |
|---|---|---|---|---|
| mistralai/mistral-small-3.2-24b-instruct | 113 | 18/67 (27%) | 7/31 (23%) | 21/120 (18%) |
| deepseek/deepseek-v4.1-flash | 45 | 7/37 (19%) | 0/4 | 1/120 (1%) |
| qwen/qwen3.7-flash | 72 | 7/46 (15%) | 2/10 (20%) | 0/120 (0%) |
| anthropic/claude-haiku-5.5 | 213 | 30/113 (27%) | 15/53 (28%) | 104/120 (87%) |

## Notes

- **Haiku's 2 referee failures**: one reply was written as prose instead of XML. Check the referee failure rate in the pilot.
- **Haiku was tested on the newer `main`** (updated prompts and game pacing). A Mistral rerun on the same code reached MAUR 30% (36/120), so the gap is smaller than the table suggests but still large.
- **Mistral's 3 referee failures were correct decisions** wrapped in a ```` ```xml ```` code fence. The referee parser rejects fences, while the player parser accepts them. Allowing fences in the referee parser would make Mistral 0/20.
- **STALLING at event 1** was the only Qwen referee failure, and gpt-oss-120b made the same call at event 1. This may be a referee prompt issue at the opening check-ball rather than a model issue.
- **DeepSeek v4-flash's invalid actions** were one repeated mistake: defenders kept proposing DENY right after their team gained possession. v4.1-flash did not repeat it.
- **Reasoning matters most.** The earlier free Nemotron run with reasoning on averaged about 5,100 output tokens per player call, versus about 150 here.
- **Speed was not compared.** Every model ran at about 14 to 19 s per event because of `max_requests_per_minute: 18` in `configs/llm.yaml`. At that limit, a 280-event game takes about 1.5 hours.
- **Referee rule search used a hash index.** The script's referee searches a hash-embedded copy of `data/rules/chunks.jsonl` instead of the sentence-transformers index. Every candidate used the same index, so the comparison is fair.
- **Small sample.** 20 events is about 7% of a game, and coordination counts are small. Re-check parse rate, IAR and the coordination metrics in the pilot.
- The 280-events-per-game figure is the plan's estimate. Real LLM games may be longer, which raises the per-game cost proportionally.

## Reproduce

```
python scripts/model_selection.py --model anthropic/claude-haiku-5.5 --events 20 --reasoning off
```

The script caps total spend across runs at $0.90 by default (`--cap`), checked against the OpenRouter key before each run.
