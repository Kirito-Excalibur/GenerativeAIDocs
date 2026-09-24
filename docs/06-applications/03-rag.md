# Retrieval-Augmented Generation

> **Summary** — Give the model the facts it needs at inference time instead of hoping they were
> memorized during training. RAG is the standard architecture for grounding LLMs in private or
> current data. This page covers the pipeline stage by stage, the chunking decisions that dominate
> quality, a failure taxonomy that tells you *which* stage to fix, and how to evaluate each stage
> separately.

**Prerequisites**: → [Embeddings & vector search](02-embeddings-and-vector-search.md) · **Next**: → [Agents & tool use](04-agents-and-tool-use.md)

---

## 1. The pipeline

```
  INDEXING (offline, once)
  ────────────────────────
  documents ──► parse ──► chunk ──► embed ──► vector index
                                      │
                                      └──► also: BM25 index, metadata store

  QUERYING (online, per request)
  ──────────────────────────────
  user query ──► [rewrite] ──► retrieve (hybrid) ──► rerank ──► top-k chunks
                                                                    │
                                                                    ▼
                          LLM ◄── prompt = system + chunks + query
                           │
                           ▼
                        answer + citations
```

> [!TIP]
> **Why RAG rather than fine-tuning.** Fine-tuning changes *behaviour*; retrieval changes
> *knowledge*. Fine-tuning a model on your documents teaches it their style and makes it hallucinate
> confidently in that style. Retrieval puts the actual text in front of it.

| | RAG | Fine-tuning |
|---|---|---|
| Update knowledge | re-index (minutes) | retrain (days) |
| Attribution | ✅ natural — cite the chunk | ❌ impossible |
| Access control | ✅ filter at retrieval time | ❌ baked into weights |
| Cost per query | higher (more tokens) | lower |
| Teaches format/style | ❌ | ✅ |
| Hallucination risk | lower (grounded) | higher |

---

## 2. Chunking: the decision that matters most

> [!WARNING]
> **Chunking quality sets the ceiling for everything downstream.** No amount of retrieval or
> reranking sophistication recovers from chunks that split a table in half or separate a claim from
> its qualifier.

```
   TOO SMALL (128 tokens)            TOO LARGE (2000 tokens)

   ┌──────────┐                      ┌────────────────────────┐
   │ "...the  │  ← context lost      │ topic A                │
   └──────────┘                      │ topic B   ← the query  │
   ┌──────────┐                      │ topic C     only needs │
   │ rate is  │  ← dangling          │ topic D     topic B    │
   └──────────┘                      │ topic E                │
   ┌──────────┐                      └────────────────────────┘
   │ 4.5%..." │                      embedding is a blurred average;
   └──────────┘                      retrieval precision drops
```

**Strategies, in increasing order of quality and effort:**

| Strategy | How | When |
|---|---|---|
| Fixed-size | $N$ tokens, $M$ overlap | baseline; always works |
| **Recursive** | split on `\n\n` → `\n` → `. ` → ` ` until under the limit | ⭐ good default |
| **Structural** | split on Markdown headings, HTML sections, code functions | ⭐ best when structure exists |
| Semantic | split where consecutive-sentence embedding similarity drops | expensive, modest gain |
| **Proposition** | an LLM rewrites text into standalone factual statements | expensive, high quality |
| Late chunking | embed the whole document, then pool per chunk | keeps global context in each chunk embedding |

**Starting parameters**: 512 tokens with 50–100 tokens of overlap. Overlap prevents a fact from
being severed at a boundary.

> [!TIP]
> **Three techniques that punch above their weight:**

**(a) Contextual retrieval** — prepend an LLM-generated context sentence to each chunk before
embedding it:

```
  Original chunk:  "Revenue grew 3% over the prior quarter."
  Contextualized:  "This chunk is from ACME Corp's Q2 2024 10-Q, discussing
                    segment performance. Revenue grew 3% over the prior quarter."
```

Anthropic reported this substantially reduces retrieval failures. It costs one cheap LLM call
per chunk at index time — a one-off expense for a permanent gain. It directly fixes the "this
chunk is meaningless without knowing what document it came from" problem.

**(b) Parent-document retrieval** — embed small chunks for precision, but return their larger
parent section to the LLM. Best of both: sharp retrieval, complete context.

**(c) Multi-representation indexing** — index a *summary* or *hypothetical question* for each
chunk, but return the chunk itself. Queries look more like questions than like document prose, so
matching question-to-question works better than question-to-prose.

---

## 3. Query processing

The user's query is often a poor retrieval key.

| Technique | Problem it solves |
|---|---|
| **Query rewriting** | conversational references ("what about last year?" → a standalone query) |
| **Multi-query** | generate 3–5 paraphrases, retrieve for each, union the results |
| **HyDE** | generate a *hypothetical answer*, embed **that**, and retrieve with it |
| Decomposition | split a multi-hop question into sub-questions |
| Step-back | ask a more general question first, retrieve background, then the specific one |
| Routing | choose which index/collection to search |

> [!TIP]
> **HyDE (Hypothetical Document Embeddings) is the cleverest of these.** Queries and documents live
> in different linguistic registers — a question ("what causes X?") looks nothing like an answer
> passage ("X is caused by..."). Asking the LLM to *hallucinate* a plausible answer and embedding
> that puts the search vector in document-space, where the real answers are. The hallucination's
> factual content is irrelevant; only its shape matters.

> [!WARNING]
> Each of these adds latency and cost. Multi-query with 5 variants means 5× the retrieval work
> plus an LLM call. Measure whether it helps on *your* data before adopting it.

---

## 4. The failure taxonomy

This table is the most practically useful thing on this page. When RAG gives a bad answer,
diagnose *which stage* failed:

| # | Failure | Symptom | Fix |
|---|---|---|---|
| 1 | **Missing content** | the answer isn't in the corpus at all | ingest more; detect and say "I don't know" |
| 2 | **Missed the top-k** | the right chunk exists but wasn't retrieved | hybrid search, better embeddings, larger $k$, reranker |
| 3 | **Lost in reranking/context** | retrieved but not used | reorder (best first/last), fewer chunks, reranker |
| 4 | **Not extracted** | the chunk is in context but the LLM missed it | better prompt, stronger model, less distraction |
| 5 | **Wrong format** | right facts, wrong shape | explicit format instruction, structured output |
| 6 | **Incomplete** | partial answer | multi-hop retrieval, larger $k$ |
| 7 | **Ignored context** | model answers from parametric memory instead | prompt to use only the context; check for conflicts |
| 8 | **Bad chunk boundaries** | truncated tables, split claims | structural chunking, larger overlap |

> [!TIP]
> **The diagnostic procedure**: for a failing query, check in order —
> (a) is the answer in the corpus? (b) is the right chunk in the top-50? (c) is it in the top-5 after
> reranking? (d) did the LLM use it? Each "no" points at exactly one fix. Debugging RAG without this
> decomposition is guesswork.

**Empirically the most common failure is #2** — retrieval recall. This is why hybrid search and
reranking are the two highest-value additions.

---

## 5. Generation

A prompt template that handles the things that actually go wrong:

```python
RAG_PROMPT = """Answer the question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say "I don't have enough information
  to answer that." Do not use outside knowledge.
- Cite the source id in brackets after each claim, like [3].
- If sources conflict, say so and present both.
- Quote exact figures and dates rather than paraphrasing them.

<context>
{context}
</context>

Question: {question}

Answer:"""

def format_context(chunks):
    # Put the most relevant chunk FIRST and the second-most LAST:
    # both ends of the context are attended to more strongly.
    if len(chunks) > 2:
        chunks = [chunks[0]] + chunks[2:] + [chunks[1]]
    return "\n\n".join(
        f"[{i}] (source: {c['source']}, section: {c.get('section','?')})\n{c['text']}"
        for i, c in enumerate(chunks, 1))
```

> [!TIP]
> **The reordering trick exploits "lost in the middle"**
> (→ [Long context §6](../04-large-language-models/09-long-context.md#6-what-long-context-actually-delivers)).
> Put your two best chunks at the extremes, where attention is strongest.

> [!WARNING]
> **Knowledge conflict** — when the retrieved context contradicts the model's parametric
> knowledge, behaviour is unpredictable. Models generally favour the context when it is clear and
> specific, and fall back on parametric memory when the context is vague. Explicit instructions
> help but are not a guarantee. For high-stakes applications, verify that cited claims actually
> appear in the cited chunk.

---

## 6. Evaluation

> [!WARNING]
> **Evaluate retrieval and generation separately.** A single end-to-end score tells you something
> is wrong but not what.

### Retrieval metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Recall@k** | (relevant in top-$k$) / (total relevant) | did we find it? ⭐ the one that matters most |
| Precision@k | (relevant in top-$k$) / $k$ | how much noise? |
| **MRR** | $\frac{1}{\|Q\|}\sum \frac{1}{\text{rank of first relevant}}$ | how high is the first hit? |
| **nDCG@k** | $\frac{\text{DCG}@k}{\text{IDCG}@k}$, $\text{DCG}=\sum_i \frac{rel_i}{\log_2(i+1)}$ | graded relevance, position-weighted |
| Hit rate | fraction of queries with ≥1 relevant hit | coarse but interpretable |

**nDCG example.** Retrieved relevance grades $[3, 0, 2, 1]$ (ideal would be $[3,2,1,0]$):

$$\text{DCG} = \frac{3}{\log_2 2} + \frac{0}{\log_2 3} + \frac{2}{\log_2 4} + \frac{1}{\log_2 5} = 3 + 0 + 1 + 0.431 = 4.431$$
$$\text{IDCG} = 3 + \frac{2}{1.585} + \frac{1}{2} + 0 = 3 + 1.262 + 0.5 = 4.762$$
$$\text{nDCG} = 4.431/4.762 = \mathbf{0.931}$$

### Generation metrics (the RAGAS framework)

| Metric | Question it answers | How |
|---|---|---|
| **Faithfulness** | are all claims supported by the context? | decompose into claims, verify each against the context |
| **Answer relevance** | does it address the question? | generate questions from the answer, compare to the original |
| **Context precision** | is the retrieved context on-topic? | judge each chunk's relevance |
| **Context recall** | does the context contain everything needed? | check the ground-truth answer's claims against the context |

> [!TIP]
> **Faithfulness is the metric to prioritize** — it directly measures hallucination, which is the
> failure mode RAG exists to prevent. A system that is 95% faithful and 80% relevant is far more
> trustworthy than the reverse.

**Building an evaluation set without human labels**: have an LLM generate questions *from* your
chunks. The source chunk is then the ground-truth relevant document by construction. This gives
you a few hundred retrieval test cases in an hour. ⚠️ It is biased toward questions that are easy
to answer from a single chunk — supplement with real user queries as soon as you have them.

---

## 7. Advanced patterns

| Pattern | Idea | When |
|---|---|---|
| **Self-RAG** | the model decides *whether* to retrieve, and critiques what it gets | mixed query types |
| **Corrective RAG (CRAG)** | grade retrieved docs; if poor, fall back to web search | open-domain |
| **GraphRAG** | build an entity/relation graph; retrieve subgraphs and community summaries | questions spanning many documents |
| **Agentic RAG** | the LLM issues retrieval as a tool call, iteratively | complex multi-hop research |
| **Recursive summarization** | hierarchical summaries (RAPTOR) | "what is this whole corpus about?" |
| **Long-context RAG** | retrieve 50 chunks instead of 5 | when the window allows it |

> [!TIP]
> **GraphRAG addresses a genuine structural gap.** Standard RAG answers "what does document X say
> about Y?" It cannot answer "what are the main themes across all 10,000 documents?" — no single
> chunk contains that, so retrieval has nothing to find. GraphRAG precomputes community summaries at
> index time so global questions have something to retrieve.

> [!TIP]
> **Long-context RAG is underrated.** Retrieval *recall* failures dominate (§4), and a big context
> window lets you trade precision for recall — retrieve 50 chunks and let the model sort it out.
> This is more robust than heroic efforts to get $k=5$ exactly right.
> → [Long context §7](../04-large-language-models/09-long-context.md#7-long-context-vs-rag)

---

## 8. A complete implementation



```python
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    source: str
    section: str = ""
    parent_id: str | None = None

class RAGPipeline:
    def __init__(self, retriever, reranker, llm):
        self.retriever, self.reranker, self.llm = retriever, reranker, llm

    def index(self, documents, chunk_size=512, overlap=64):
        chunks = []
        for doc in documents:
            for c in recursive_split(doc.text, chunk_size, overlap):
                # contextual retrieval: a one-off LLM call per chunk at index time
                ctx = self.llm(
                    f"<document>{doc.text[:4000]}</document>\n"
                    f"<chunk>{c}</chunk>\n"
                    "In one sentence, situate this chunk within the document. "
                    "Output only that sentence.")
                chunks.append(Chunk(text=f"{ctx}\n\n{c}", source=doc.id,
                                    section=doc.section))
        self.retriever.build(chunks)      # builds BOTH the dense and BM25 indexes
        self.chunks = chunks

    def query(self, question, k=5, candidates=50, filters=None):
        # 1. retrieve widely (hybrid), with metadata filters for access control
        cands = self.retriever.search(question, n=candidates, filters=filters)
        # 2. rerank with a cross-encoder
        scored = self.reranker.rank(question, cands)
        top = scored[:k]
        # 3. abstain rather than answer from irrelevant context
        if not top or top[0].score < RELEVANCE_FLOOR:
            return {"answer": "I don't have enough information to answer that.",
                    "sources": []}
        # 4. generate with citations
        answer = self.llm(RAG_PROMPT.format(
            context=format_context(top), question=question))
        return {"answer": answer,
                "sources": [{"id": i, "source": c.source, "text": c.text[:200]}
                            for i, c in enumerate(top, 1)]}
```

> [!WARNING]
> **The `RELEVANCE_FLOOR` check is the single most important line for trustworthiness.** Without
> it, a query with no good match still retrieves the five *least bad* chunks and the model dutifully
> answers from irrelevant context. Calibrate the floor on known-unanswerable queries.

> [!WARNING]
> **Metadata filters are how you do access control.** Never rely on the LLM to withhold
> information present in its context — filter at retrieval time, before anything reaches the model.

---

## 9. Key takeaways

| # | Takeaway |
|---|---|
| 1 | RAG supplies knowledge; fine-tuning shapes behaviour. Use the right tool. |
| 2 | Chunking sets the ceiling. Start at 512 tokens with overlap, and respect document structure. |
| 3 | Contextual retrieval (prepend an LLM-written situating sentence) is a cheap, large win. |
| 4 | HyDE works by moving the query into document-space; the hallucinated answer's accuracy is irrelevant. |
| 5 | Use the 8-way failure taxonomy to identify *which stage* broke before changing anything. |
| 6 | Retrieval recall is the most common failure → hybrid search + reranking are the top two fixes. |
| 7 | Put the best chunks at the start and end of the context. |
| 8 | Evaluate retrieval (recall@k, nDCG) and generation (faithfulness) **separately**. |
| 9 | Always implement an abstention path with a relevance floor. |
| 10 | Enforce access control with retrieval-time metadata filters, never by instructing the model. |

---

## Further reading

- Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) (2020) — the original.
- Gao et al., [*Retrieval-Augmented Generation for Large Language Models: A Survey*](https://arxiv.org/abs/2312.10997) (2023).
- Barnett et al., [*Seven Failure Points When Engineering a RAG System*](https://arxiv.org/abs/2401.05856) (2024).
- Gao et al., [*Precise Zero-Shot Dense Retrieval without Relevance Labels*](https://arxiv.org/abs/2212.10496) (HyDE, 2022).
- Es et al., [*RAGAS: Automated Evaluation of Retrieval Augmented Generation*](https://arxiv.org/abs/2309.15217) (2023).
- Edge et al., [*From Local to Global: A GraphRAG Approach to Query-Focused Summarization*](https://arxiv.org/abs/2404.16130) (2024).
- Sarthi et al., [*RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval*](https://arxiv.org/abs/2401.18059) (2024).

**Next** → [Agents & tool use](04-agents-and-tool-use.md)
