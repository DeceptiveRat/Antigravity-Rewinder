Both `PoC2/response.txt` and `PoC3/response.txt` were generated under identical conditions and exhibit a classic LLM phenomenon: **semantic convergence with surface and structural divergence**.

Specifically:
- **Shared semantics**: Both responses identify the exact same core themes: romantic infatuation/passion, Phil Collen's quote on spiritual enlightenment, Rick Allen's 1984 car accident context, the subversion of the word "hysteria" from panic to ecstasy, and the phrase "magical mysteria".
- **Divergence**: 
  - Structural organization: `PoC2` breaks Section 1 into 4 numbered points, while `PoC3` condenses it into 3 points.
  - Phrasing and formatting: `PoC3` adds an external markdown link and bullet points under Section 2, while `PoC2` uses single paragraphs with bold headers.

In causal inference and counterfactual evaluation, proving whether an intervention caused a divergence requires measuring how much two responses match under the null condition (identical inputs) versus under an intervention ($do(X)$).

Here are the primary methodologies to measure this match, categorized by the level of abstraction:

---

### 1. Atomic Claim Decomposition and NLI Entailment (Recommended for Counterfactual Proofs)
If the goal is to prove whether two outputs state the same facts or draw the same conclusions, surface text metrics are insufficient. Atomic claim decomposition is the standard in factuality and causal verification (e.g., FActScore or SAFE methodology).

**How it works**:
1. Decompose both texts into sets of atomic propositions:
   - Claim Set A (from PoC2): e.g., `["Phil Collen stated the song is about spiritual enlightenment", "Rick Allen suggested the title after his 1984 accident", "The song subverts hysteria into euphoric love", ...]`.
   - Claim Set B (from PoC3): e.g., `["The song operates on romantic and spiritual levels", "Phil Collen noted the track is about spiritual enlightenment", "Rick Allen lost his arm in a 1984 car crash", ...]`.
2. Evaluate bidirectional Natural Language Inference (NLI) or LLM entailment between the two claim sets:
   - **Forward Coverage (Recall)**: Percentage of claims in A entailed by B.
   - **Backward Coverage (Precision)**: Percentage of claims in B entailed by A.
   - **Claim F1 Score**: The harmonic mean of bidirectional coverage.
3. **Why it matters for counterfactuals**: If an intervention causes the agent to miss or alter specific factual claims, the Claim F1 will drop significantly below the identical-condition baseline.

---

### 2. Dense Semantic Similarity (Continuous Embedding Space)
Measures the distance between the representations in continuous vector space.

**Approaches**:
- **Document-level Embedding Cosine Similarity**: Use a text embedding model (such as `text-embedding-3-large`, `all-mpnet-base-v2`, or Vertex text-embeddings) to embed both full texts and calculate cosine similarity:
  $$\text{CosineSim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
- **BERTScore**: Token-level greedy matching using contextual embeddings. It aligns each token in PoC2 to the most similar token in PoC3 using cosine similarity, yielding Precision, Recall, and F1. Unlike sentence embeddings, BERTScore is sensitive to word-order differences and penalizes omitted concepts.

---

### 3. Structural and Markdown AST Distance
In agent evaluation, the model often keeps the same thoughts but changes its formatting or hierarchical structure.

**How it works**:
1. Parse the Markdown into an Abstract Syntax Tree (AST) using a markdown parser (e.g., `mistletoe` or `markdown-it-py`).
2. Extract the topology:
   - Header depth distribution (H1, H2, H3).
   - Number and nesting depth of list items.
   - Presence of blockquotes and external links.
3. Calculate **Tree Edit Distance** (e.g., using the APTED algorithm) or Jaccard similarity over the node sequence types.
4. **Why it matters**: Allows isolating *form* divergence from *content* divergence.

---

### 4. Surface Lexical & N-gram Overlap (Baseline Metrics)
These give a fast, deterministic baseline, though they do not understand paraphrasing:
- **ROUGE-L**: Measures the Longest Common Subsequence (LCS) to capture sentence-level structure retention.
- **Token Set Jaccard Similarity**: Measures the ratio of shared unique tokens or bigrams.
- **Normalized Levenshtein / Character Edit Distance**: Captures character-level similarity ratio ($1 - \frac{\text{edit\_dist}}{\max(\text{len}_1, \text{len}_2)}$).

---

### 5. Multi-Axis LLM-as-a-Judge Evaluation (G-Eval Rubric)
Using a prompt-based judge model to evaluate the pairwise match across decoupled dimensions on a normalized scale (0.0 to 1.0):
- **Core Semantic Equivalence**: Do both outputs express the same analytical conclusions?
- **Factual & Citation Overlap**: Do both cite the same evidence and quotes?
- **Rhetorical / Stylistic Alignment**: Are the tone, phrasing, and verbosity comparable?
- **Action / Decision Alignment**: If applied to agent workflows (e.g., tool calls), did both decide on identical actions?

---

### How to Use This for Counterfactual Proofs

To establish a valid counterfactual proof:
1. **Define the Null Distribution ($H_0$)**:
   Run multiple trials under the exact same prefix/condition (e.g., `PoC1`, `PoC2`, `PoC3`). Compute pairwise distance $D(Y_i, Y_j)$ across your chosen metric (e.g., Claim F1 or Semantic Embedding Distance). This measures the natural stochastic variance (the temperature/sampling jitter).
2. **Measure the Intervention Distance ($H_1$)**:
   Apply an intervention at a prior step ($do(X)$), and compute $D(Y_{\text{factual}}, Y_{\text{counterfactual}})$.
3. **Statistical Hypothesis Test**:
   Prove that $D_{\text{intervention}} > D_{\text{stochastic}}$ with statistical significance (e.g., via bootstrap permutation testing or Welch's t-test). If the distance under intervention falls within the $H_0$ distribution, the intervention cannot be proven to have caused a causal shift.

---

### Clarifying Questions

To help select or implement the exact method for your experiment:
1. What specific aspect of the divergence are you trying to prove or control? (e.g., semantic/factual consistency, downstream tool action selection, or surface format stability?)
2. Are you evaluating free-form natural language responses, or also structured tool call invocations and arguments?
3. What scale or volume of counterfactual runs do you plan to measure?