# Structured Output and Constrained Decoding

> **Summary** — Getting an LLM to emit valid JSON, a specific schema, or any formal grammar — with
> a *guarantee*, not a hope. The mechanism is simple: compile the grammar into a finite-state
> machine and mask the logits of every token that would violate it. This page covers how that
> works, its cost, the quality trade-offs constraints introduce, and how to design schemas that
> don't fight the model.

**Prerequisites**: → [Inference & decoding](../04-large-language-models/06-inference-and-decoding.md) · **Next**: Part VII → [Evaluation metrics](../07-evaluation/01-metrics.md)

---

## 1. The three levels of guarantee

```
  LEVEL 1: ASK NICELY                      ~85-95% valid
  "Respond with JSON matching this schema: {...}"
  ⚠️ fails on: markdown fences, trailing commas, explanatory preamble,
     truncation, single quotes, comments

  LEVEL 2: ASK + PARSE + RETRY             ~99% valid
  try: json.loads(out)
  except: send the parse error back and retry
  ⚠️ costs extra latency and tokens; can still fail

  LEVEL 3: CONSTRAINED DECODING            100% valid, by construction
  mask invalid tokens before sampling — an invalid string is
  literally unreachable
```

> [!TIP]
> **Level 3 is a different kind of thing.** Levels 1 and 2 reduce the probability of failure.
> Level 3 makes failure *impossible* — it is a correctness guarantee in the same sense a type system
> is. If your downstream code parses the output, this distinction matters.

---

## 2. How constrained decoding works

```
   Schema: {"name": string, "age": integer}

   generated so far:  {"name": "Ada", "age":
                                             ▲
                        the FSM says: only whitespace or a digit is valid here

   logits:    "5"  12.3  ──► allowed
              " "   8.1  ──► allowed
              "x"  15.7  ──► set to −∞   ← would be the argmax, but it's invalid!
              "}"  11.2  ──► set to −∞
              ...
                    ▼
              softmax over the survivors ──► sample
```

**The algorithm:**

1. Compile the schema/grammar into a **finite-state machine** (or a pushdown automaton for
   recursive grammars).
2. For each FSM state, precompute a **bitmask** over the vocabulary: which tokens can legally
   come next.
3. At each decoding step, look up the mask for the current state, set disallowed logits to
   $-\infty$, sample, and advance the FSM.

**The cost is near zero at runtime** because step 2 is done once, offline. The expensive part is
compilation, which is cached per schema. Modern implementations (XGrammar, Outlines) report
overhead of a few percent or less.

> [!WARNING]
> **The subtlety: tokens vs characters.** The grammar is defined over characters, but the model
> emits *tokens*, which span multiple characters. A token like `":"` or `", "` or `"}]}"` may advance
> the FSM several states at once, and a token may be *partially* valid (valid prefix, invalid
> continuation). Handling this correctly — building the token-level FSM from the character-level one
> — is the actual engineering content of these libraries.

---

## 3. What you can constrain

| Constraint | Machinery | Example |
|---|---|---|
| **JSON Schema** | FSM / pushdown automaton | `{"type":"object","properties":{...}}` |
| Regular expression | FSM | `\d{3}-\d{2}-\d{4}` |
| **Context-free grammar** | pushdown automaton | SQL, a DSL, valid Python |
| Enum / choice | trivial FSM | `"positive" \| "negative" \| "neutral"` |
| Type coercion | FSM | integer, float, boolean, ISO date |

The common libraries:

```python
# --- Outlines: regex, JSON schema, or a Pydantic model ---
from outlines import models, generate
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    vendor: str
    total: float = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    line_items: list[str]

model = models.transformers("meta-llama/Llama-3.1-8B-Instruct")
generator = generate.json(model, Invoice)
result: Invoice = generator("Extract the invoice: ...")   # a typed object, guaranteed

# --- llama.cpp: GBNF grammars ---
# root   ::= "{" ws "\"answer\":" ws string ws "}"
# string ::= "\"" ([^"\\] | "\\" .)* "\""

# --- vLLM / SGLang: guided decoding in the serving layer ---
# response_format={"type": "json_schema", "json_schema": {...}}
```

**API-level structured output** (OpenAI's Structured Outputs, Anthropic's tool-use schemas,
Gemini's response schema) is the same mechanism exposed as a parameter. If your provider offers it,
use it — you get the guarantee without running your own inference stack.

---

## 4. The quality trade-off

> [!WARNING]
> **Constraints guarantee *syntax*, not *semantics*.** A model forced into a schema it finds
> unnatural will produce valid JSON containing worse content.

```
  Unconstrained:                      Over-constrained:
  "This contract has a concerning     {"risk": "medium", "clause": "8.3"}
   clause in 8.3 — it allows           ▲ forced to pick from an enum before
   unilateral repricing, which is        it had the chance to reason about it
   high risk in a 5-year term..."
```

> [!TIP]
> **Why this happens.** Constrained decoding masks tokens the model *wanted* to emit. If the
> model's natural high-probability continuation was "Let me look at clause 8.3 first...", masking
> that forces it into a lower-probability region of its distribution — exactly where its outputs are
> less reliable.

**The fix that reliably works: give the model a place to think first.**

```json
{
  "type": "object",
  "properties": {
    "reasoning": {"type": "string",
                  "description": "Analyze the evidence before deciding."},
    "risk": {"type": "string", "enum": ["low", "medium", "high"]},
    "clause": {"type": "string"}
  },
  "required": ["reasoning", "risk", "clause"]
}
```

> [!WARNING]
> **Field order matters**, because generation is autoregressive. Put `reasoning` **first** so the
> classification is conditioned on it. Put it last and it is a post-hoc rationalization with no
> influence — the same principle as
> → [chain-of-thought ordering](01-prompt-engineering.md#ask-for-reasoning-before-the-answer).

> [!TIP]
> **JSON Schema's `properties` order is not guaranteed to be honoured by every implementation.**
> Check that your library generates fields in declaration order; some do not, and the reasoning
> field silently stops helping.

---

## 5. Schema design principles

| ✅ Do | ❌ Don't |
|---|---|
| Put a reasoning/explanation field **first** | put it last, or omit it |
| Use `enum` for closed sets | use a free string and post-process |
| Add `description` to every field — it's a prompt | leave fields unexplained |
| Provide a "not found" / "unknown" value | force a choice when the answer is absent |
| Keep nesting shallow (≤3 levels) | build deeply nested trees |
| Use flat arrays of objects | use objects with dynamic keys |
| Set `"additionalProperties": false` | allow arbitrary extra keys |
| Make optional things genuinely optional | mark everything `required` |

> [!TIP]
> **The "not found" value is the most commonly missed one.** A schema requiring
> `{"price": number}` forces the model to emit *some* number even when the document contains no
> price. Add `{"price": {"type": ["number", "null"]}}` and an explicit instruction, and extraction
> accuracy on absent fields goes from ~0 to near-perfect. **You cannot constrain your way out of
> having asked the wrong question.**

> [!TIP]
> **Dynamic keys are a trap**: `{"Paris": 5, "Tokyo": 3}` cannot be expressed as a fixed schema and
> forces the model into an open structure. Use `[{"city": "Paris", "count": 5}, ...]` instead —
> constrainable, and easier to validate.

---

## 6. Tool calling is structured output

The same machinery, different framing. A tool's `input_schema` is a JSON Schema, and the model's
tool call is constrained to match it.

```
  Tool definition                  Model output (constrained)
  ───────────────                  ──────────────────────────
  name: get_weather                {"name": "get_weather",
  schema: {                         "input": {"location": "Paris, France",
    location: string (required)               "unit": "celsius"}}
    unit: enum[celsius, fahrenheit]
  }                                ✅ guaranteed to parse
                                   ✅ guaranteed to have `location`
                                   ✅ `unit` guaranteed to be a valid enum value
```

**This is why modern tool calling is so much more reliable than 2023-era "output a JSON blob"
prompting.** The failure mode "the model produced malformed arguments" is structurally eliminated,
which directly raises per-step agent reliability — and per
→ [Agents §4](04-agents-and-tool-use.md#4-the-compounding-error-problem), per-step reliability is
what determines whether long-horizon tasks work at all.

---

## 7. When constrained decoding isn't available

Not every provider or model exposes it. The fallback stack:

```python
import json, re
from pydantic import BaseModel, ValidationError

def extract_json(text: str) -> str:
    """Salvage JSON from a response with markdown fences or preamble."""
    # strip ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    # take the outermost balanced {...} or [...]
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("no JSON found")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:       esc = False
            elif ch == "\\": esc = True
            elif ch == '"':  in_str = False
        elif ch == '"':  in_str = True
        elif ch in "{[":  depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced JSON")

def robust_parse(model, prompt, schema: type[BaseModel], max_retries=2):
    for attempt in range(max_retries + 1):
        # prefill the assistant turn — the cheapest format enforcement available
        out = model(prompt, prefill="{" if attempt == 0 else "")
        try:
            return schema.model_validate_json(extract_json(out))
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries:
                raise
            # feed the exact error back; models are good at fixing their own output
            prompt += (f"\n\nYour previous output was invalid:\n{out}\n"
                       f"Error: {e}\nOutput ONLY valid JSON matching the schema.")
```

**Prefilling `{`** (level 1.5) costs nothing and eliminates the most common failure — the model
prefacing its JSON with "Sure! Here's the JSON:". Combined with retry-on-error, this reaches ~99%
without any special inference support.

---

## 8. Exercises

**Problem 1 — the three levels, applied.** A team currently uses Level 1 ("ask nicely") for
extracting structured data from invoices and observes a 91% valid-JSON rate. Using §1's level
descriptions, roughly how much improvement would moving to Level 2 (ask + parse + retry) versus
Level 3 (constrained decoding) buy them, and which level *guarantees* zero malformed-JSON
failures downstream?

<details markdown="1"><summary>Solution</summary>

Per §1: Level 1 sits at "~85–95%" (their 91% fits squarely in this range). Level 2 reaches
"~99%" — a meaningful jump, cutting the failure rate roughly from ~9% to ~1% (about a 9× 
reduction in the specific case of 91%→99%). Level 3 reaches "100%, by construction" — not "very
close to 100%," but an actual guarantee, because invalid strings are *structurally unreachable*
(§2), not merely made improbable.

**Only Level 3** guarantees zero malformed-JSON failures downstream — Level 2's 99% still means
roughly 1-in-100 requests could hit a parsing failure that either needs a fallback path or causes
a visible error, which matters for a fully automated invoice pipeline with no human review step.

</details>

**Problem 2 — schema design, spot the problem.** A schema for extracting contract terms has:
`{"party_a": string, "party_b": string, "effective_date": string, "termination_date": string}`,
all marked required. Using §5's principles, identify two design flaws and their likely
consequence.

<details markdown="1"><summary>Solution</summary>

**Flaw 1 — no "not found" option** (§5: "provide a 'not found' / 'unknown' value"). Not every
contract has an explicit termination date (some are open-ended); marking `termination_date` as
required forces the model to invent a plausible-sounding date when none exists in the source
document — a direct hallucination risk that §5 calls "the most commonly missed" schema mistake.

**Flaw 2 — no reasoning field, and it's positioned wrong even if added naively** (§4: "put the
reasoning/explanation field **first**"). Extracting `effective_date` vs `termination_date`
correctly from a contract that mentions multiple dates in different clauses is exactly the kind
of task that benefits from the model working through *which* date is which before committing to
an answer — without a first-positioned reasoning field, the schema forces the model straight into
extraction with no room to disambiguate, which is likely to increase the error rate on contracts
with several candidate dates.

</details>

**Problem 3 — field order and autoregressive conditioning, why it matters here specifically.**
§4 says field order matters because of autoregressive generation. A colleague argues "the model
sees the whole schema up front anyway (it's in the prompt), so by the time it starts generating
the `risk` field, it already 'knows' what `reasoning` will eventually say, regardless of which
field comes first in the *output*." What's wrong with this argument?

<details markdown="1"><summary>Solution</summary>

The colleague conflates *seeing the schema* (which is indeed available up front, in the prompt)
with *having already computed the reasoning's content*. The schema only specifies field
**names** and types — it doesn't contain the actual reasoning text, which the model must
generate token-by-token as part of its output. Per §4's "autoregressive order determines what
conditions what": if `risk` is generated *before* `reasoning` in the output sequence, then when
the model is producing the `risk` value, the specific words of the (not-yet-written) reasoning
don't exist yet as tokens the model can condition on — only the abstract fact that a reasoning
field *will* follow is "known" (from the schema), not its content. The classification is
therefore made without the benefit of having worked through the reasoning, exactly the same
issue as putting a chain-of-thought *after* the final answer in ordinary prompting
(→ [Prompt engineering, "reasoning before the
answer"](01-prompt-engineering.md#ask-for-reasoning-before-the-answer)) — knowing a scratchpad
section will eventually be filled in is not the same as having its contents to condition on.

</details>

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | Three levels: ask (85–95%), ask+retry (~99%), constrained decoding (**100%, by construction**). |
| 2 | The mechanism is an FSM over the grammar + a logit mask; runtime cost is near zero. |
| 3 | The hard engineering is reconciling character-level grammars with multi-character tokens. |
| 4 | Constraints guarantee syntax, not semantics — a forced schema can *lower* content quality. |
| 5 | Put a `reasoning` field **first** in the schema; autoregressive order determines what conditions what. |
| 6 | Always provide a null / "not found" option, or the model will invent values. |
| 7 | Avoid dynamic keys and deep nesting; use flat arrays of objects. |
| 8 | Field `description`s are prompts — write them as such. |
| 9 | Tool calling is constrained decoding, which is why modern agents fail far less on argument formatting. |
| 10 | Without native support: prefill `{`, salvage from fences, validate, and retry with the error. |

---

## Further reading

- Willard & Louf, [*Efficient Guided Generation for Large Language Models*](https://arxiv.org/abs/2307.09702) (Outlines, 2023).
- Dong et al., [*XGrammar: Flexible and Efficient Structured Generation*](https://arxiv.org/abs/2411.15100) (2024).
- Geng et al., [*Grammar-Constrained Decoding for Structured NLP Tasks*](https://arxiv.org/abs/2305.13971) (2023).
- Tam et al., [*Let Me Speak Freely? A Study on the Impact of Format Restrictions on LLM Performance*](https://arxiv.org/abs/2408.02442) (2024) — the quality trade-off, measured.
- The JSON Schema specification — `json-schema.org`.

**Next** → Part VII: [Evaluation metrics](../07-evaluation/01-metrics.md)
