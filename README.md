# Vision-Transformers: Crafting a Task-Agnostic Universal Adversarial Perturbation by Targeting Evaluation Through Psychological and Cognitive Biases

This project studies a large vision-language model (**Llama-4-Scout-17B-16E-Instruct**)
as an *evaluator* — a system asked to judge images (how interesting they are,
whether they're relevant, how it "feels" about them) — and asks whether that
evaluative judgement rests on **linear, manipulable internal directions**
tied to persona conditioning (gender, emotional state, cultural background)
and affective state (valence/arousal).

The end goal is a **single, task-agnostic image perturbation**: one universal
adversarial delta that, added to *any* image, shifts the model's evaluation of
it — not by exploiting a specific prompt or output vocabulary, but by pushing
the model's internal state along a **psychologically-grounded axis** (e.g.
"more awe/excitement-coded", "more engaging") that the model's evaluative
judgement reads from. If it works, the attack should transfer across
differently-worded evaluation tasks, because it targets the underlying
cognitive/affective state rather than a task-specific shortcut.

The guiding question:

> Does the model represent affective/persona-conditioned judgement as a
> **linear direction** in its hidden activations? Is that direction **causally
> responsible** for its evaluative outputs? And if so, can it be **reached
> from pixel space alone** — a perturbation baked into an image, requiring no
> access to prompts or activations at inference time — to produce a
> perturbation that biases evaluation broadly, across tasks, by exploiting the
> same underlying psychological bias?

---

## A five-tier study

| Tier | Question | Status |
|---|---|---|
| **I — Representation Discovery** (`representation/`) | Do persona/affect contrasts correspond to consistent linear directions in activation space, and where (which layer) are they strongest? | done |
| **II — Analytics** (`analytics/`) | What do those directions mean geometrically — are they shared across conditions, do they cluster, do they compose additively? | in progress |
| **III — Causal Control** (`representation/III1_vector_control/`, `control/`) | Does intervening on a direction in the residual stream (inject / scale / ablate) actually change the model's output, and in the predicted way? | in progress |
| **IV — Universal Adversarial Perturbation** (`attack/IV1_gradient_matching/`) | Can a single pixel-space perturbation, trained against a validated direction, shift evaluation on the task it was trained against — i.e. is the attack semantic rather than token-level? | done |
| **IV.2 — Generalisation Level 1** (`attack/IV2_generalisation/`) | Do those trained UAPs transfer to *genuinely unseen datasets and downstream tasks*, beyond the interestingness/relevance framings they were trained/first evaluated on? | done |

Tiers I-III are the representational groundwork: find candidate directions,
understand their geometry, and confirm which ones are actually load-bearing
for the model's judgement (as opposed to merely decodable). Tier IV is the
attack itself — it takes whichever direction Tier III validates as causally
potent and asks whether that causal effect can be induced purely through an
image perturbation, crafted once and applied universally. Tier IV.2 pushes
that transfer question further out-of-domain: the same trained UAPs, unmodified,
evaluated against three new datasets and prompt framings that have nothing to
do with the original persona-interestingness setup.

---

## Status

The representational groundwork (Tiers I-III) has produced working evidence
that some persona/affect directions are both decodable and causally
load-bearing for the model's evaluative output, while others (e.g. purely
identity-coded directions) are decodable but do not move the evaluation — a
distinction that directly shaped which directions Tier IV targets.

### Tier IV — Universal Adversarial Perturbation

Three UAPs were trained, each a single pixel-space delta (L∞ ball, swept over
ε ∈ {0.1, 0.5, 1.0, 2.0}) targeting one validated direction at layer
`language_29` (`language_24` for an earlier workload variant, superseded by a
`language_29` run for cross-attack comparability):

- **interest** — blank-prompt interestingness axis
- **excited_vs_angry** — excitement-vs-anger affect axis
- **workload** (aka "stress") — mental-workload/overwhelm axis

All three were first evaluated on the tasks they were built from — an
interestingness rating task and a binary relevance task
(`results/attack_eval_projected/{interestingness,relevance}/`). Both show the
attack is behaviorally real (large rating shifts at higher ε) but also prone
to saturation: at ε=2.0, interestingness ratings collapse to "Extremely
Interesting" for 100% of images under both `interest` and `excited_vs_angry`.

### Tier IV.2 — Generalisation Level 1

The three trained UAPs (no retraining, no scaling — used exactly as saved)
were evaluated against three new out-of-domain tasks, each on a frozen,
stratified sample with a clean baseline paired per-image against every
perturbed condition:

| Task | Dataset | n | Judgment |
|---|---|---|---|
| **Shopping relevance** | Marqo-GS-10M | 300 | 1-5 relevance of a product image to a shopping query |
| **Moral evaluation** | SMID | 274 | 1-5 morality of a photographed scene |
| **Damage severity** | QCRI/MEDIC | 300 | 0-2 visible disaster-damage severity |


**Headline findings:**

- **All three UAPs generalise** — none is a token-level artifact specific to
  the persona-interestingness prompt. Every attack produces a significant,
  budget-dependent shift on every one of the three new tasks.
- **At low-to-moderate ε (0.1-1.0)**, the effect is modest and largely
  *coherent*: the model still recognises the actual image content, just with
  a systematic bias in its judgement.
- **At ε=2.0, behaviour is dominated by attack-specific hallucination rather
  than a graded semantic push.** Each UAP overrides the image with its own
  fixed hallucinated content, and the downstream label is just whatever that
  content implies for the specific question being asked:
  - `excited_vs_angry` → a colorful costume/festival scene (near-identical
    wording across unrelated images) → reads as low damage, morally neutral
    on the moral task, and off-topic ("not relevant") on shopping.
  - `workload` → distorted, often-profane-looking text → reads as high
    damage, immoral, and off-topic on shopping.
  - `interest` → an unstable mix of "vibrant abstract art" and
    "torn/fragmented" content → the least consistent transfer of the three,
    frequently reversing direction at ε=2.0.
- **`workload` shows the cleanest, most monotonic budget-response** of the
  three attacks on both damage severity (Spearman ρ=1.0, p=0.0) and moral
  evaluation (ρ=−1.0, p=0.0); `interest` is the least monotonic on both.
- **The same UAP can flip the sign of its effect depending on how the task
  defines the judgment**, not on anything different the attack itself is
  doing: `excited_vs_angry` pushed the *original*, ungrounded, self-referential
  relevance task ("is this relevant to you?") toward "yes", but pushes the
  new, query-grounded shopping task ("is this relevant to *this specific
  query*?") toward "no" — because the same hallucinated scene almost never
  matches an arbitrary shopping query, whereas a vivid/engaging hallucination
  satisfies an ungrounded "relevant to me" judgment by default.
- The SMID arousal analysis found a significant positive interaction between
  independently human-rated arousal and ε for all three attacks (p<1e-4), i.e.
  susceptibility to all three UAPs grows with an image's normative arousal as
  budget increases — most strongly for `workload` (R²=0.59 vs ≈0.34 for the
  other two in a pooled `delta_morality ~ arousal × epsilon + clean_morality`
  regression).

Generalisation Level 2 (amplitude/scaling dependence of a transferred attack)
is explicitly out of scope for this round and not yet started.

---

## References

- Turner et al., **ActAdd: Steering Language Models Without Optimization**, arXiv:2308.10248
- Kim et al., **TCAV: Interpretability Beyond Feature Attribution**, arXiv:1711.11279
- Zou et al., **Representation Engineering: A Top-Down Approach to AI Transparency**, arXiv:2310.01405
- Tigges et al., **Language Models Linearly Represent Sentiment**, BlackboxNLP @ ACL 2024
- Park et al., **The Linear Representation Hypothesis and the Geometry of Large Language Models**, arXiv:2311.03658
- Rimsky et al., **Steering Llama 2 via Contrastive Activation Addition**, arXiv:2312.06681
- Chen et al. (Anthropic), **Persona Vectors: Monitoring and Controlling Character Traits in Language Models**, arXiv:2507.21509
- nostalgebraist, **interpreting GPT: the logit lens**, LessWrong 2020
- **Activation Scaling for Steering and Interpreting Language Models**, EMNLP 2024 Findings (aclanthology 2024.findings-emnlp.479)
- Kriegeskorte et al., **Representational Similarity Analysis**, Frontiers in Systems Neuroscience 2008
- Kornblith et al., **Similarity of Neural Network Representations Revisited (CKA)**, arXiv:1905.00414
