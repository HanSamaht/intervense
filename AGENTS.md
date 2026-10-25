# Intervense Project Rules (Mandatory for all AI agents)

## Absolute rule: English only

ALL files must contain ZERO Chinese characters. This includes code, comments, docstrings, variable names, string literals, test assertions, README, documentation, configuration files, concept packs. No exceptions. No Chinese punctuation either. Before any commit, verify zero Chinese. If any Chinese is found, the commit is blocked.

## Writing style

- No exclamation marks
- No em dashes or en dashes. Use colons, commas, or periods.
- No dramatic language
- No rhetorical questions
- No emotional appeals
- No unnecessary bold text
- State facts plainly: subject, verb, object, number
- Let numbers speak. Do not interpret them for the reader.
- Remove filler words
- One sentence says one thing
- Write like a technical report
- No AI-sounding phrases: avoid "delve", "crucial", "landscape", "leverage",
  "furthermore", "moreover", "it is worth noting", "notably"
- No metaphors or colorful language; state facts and implications plainly
- No rhetorical questions
- No dramatic transitions ("But here is the thing", "The key insight is")
- Prefer short declarative sentences over compound constructions
- Numbers and evidence replace adjectives; never write "significant" without a p-value
- Confidence is expressed by the strength of the number, not by intensifiers

## Bug logging

1. Every error, bug, test failure must be logged to BUG_REGISTRY.md immediately
2. Log entry before fixing; update after resolution
3. Any finding must be tagged: framework tool output / manual analysis / third-party source
4. Never skip test regression
5. Every error comes with an explicit apology to the project owner before the fix: state the mistake, say sorry, then fix it. Acknowledgement without the apology does not count

## Version discipline

1. Every pushed change bumps the version: patch for fixes and cleanup, minor for features
2. Bump in both pyproject.toml and intervense/__init__.py `__version__`
3. The commit message states the new version

## Test layering

1. End-to-end tests that train demo models carry `pytest.mark.slow`
2. Push CI runs `-m "not slow"` (fast suite, under 2 minutes)
3. The nightly workflow runs the full suite and the Kaggle artifact smoke
4. A new slow test needs a fast unit companion or a justification in the commit

## Competition deliverable rule (added 2026-10-24, user mandate)

For competition entries the owner wants the strongest single-shot
model delivered directly. No staged validation ceremonies, no interim
versions presented as deliverables, no process narrated as progress.
Decision-layer tuning happens internally before delivery, not as a
reported step.

## Other

- API keys only in .env
- Diagnostic output to stderr
- Large data never enters LLM conversation
- Evidence graded: interventional / observational / prior / manual
- Audit only, do not prevent
- License: AGPL-3.0

## Kaggle push discipline (added 2026-10-23, user mandate)

1. NEVER push, submit, upload, or publish anything to Kaggle (kernels,
   datasets, notebooks, forum posts, competition entries) without an
   explicit go-ahead from the project owner in the current
   conversation. Prior approval of one action never covers the next.
2. Do not even use the word "submit" in replies. Describe the pending
   action plainly (for example: "ready to push kernel v8") and wait.
3. This rule overrides any autonomy instruction elsewhere in this file.
