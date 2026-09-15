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
| **II — Analytics** (`analytics/`) | What do those directions mean geometrically — are they shared across conditions, do they cluster, do they compose additively? | done |
| **III — Causal Control** (`control/III1_vector_control/`) | Does intervening on a direction in the residual stream (inject / scale / ablate) actually change the model's output, and in the predicted way? | done |
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

## Findings

**Central finding: yes.** Human-recognisable psychological failure modes —
specifically, the difference between *identity* (who someone is) and
*affective/arousal state* (how keyed-up someone is) as drivers of a
judgement — are present in this model as literal linear structure, and the
state-coded ones are a genuine, causally-validated attack surface that
transfers across tasks in a way a label-level attack has no reason to. This
isn't asserted from the final generalisation numbers alone — every tier
below independently narrows in on the same conclusion, and each one found
something that shaped the next.

### Tier I — the categories exist, cleanly, and only in the language stream

For every persona/affect contrast tested, a simple mean-difference direction
is close to perfectly linearly decodable, but *where* varies systematically
by category:

| Contrast family | Best layer(s) | Peak AUC | Peak cos(mean-diff, probe) |
|---|---|---|---|
| Gender (8 emotion-matched pairs) | `language_29`/`30` | **1.0** | 0.98-0.99 |
| Country/culture, Nigeria variant | `language_23`-`28` | **1.0** | ~0.96 |
| Country/culture, Germany variant | scattered, `language_2`-`30` | 0.997-1.0 | 0.91-0.95 (noisier) |
| Emotion, one-vs-rest (8 emotions) | `language_22`-`32` | 0.92-1.0 | often under 0.7 (`sad`: 0.48) |
| Emotion, vs. a contentment anchor | `language_23`/`24` | **1.0** | 0.95-0.98 |
| Mental workload / stress | `language_24`-`29` | **1.0** | 0.97-0.98 |
| Interestingness (the judgement itself) | `language_27`-`38+` | **1.0** | up to 0.96 |

Three patterns recur across every one of these ~90 contrasts:

- **Identity peaks latest and sharpest** (gender/country cluster tightly
  around `language_29`); **affect peaks earlier and broader** (emotion/
  workload, `language_22`-`29`); **the evaluative judgement itself peaks
  latest and broadest of all** (interestingness stays near-ceiling across
  17-22 of 48 language layers, and the persona-*pooled* "global" interest
  direction doesn't peak until the high-30s).
- **Every one of these directions lives in the language stream only.**
  Vision-tower layers never clear ~0.50-0.56 AUC (chance) for gender,
  emotion, country, or workload — these are injected-persona-text concepts
  with no pixel correlate, so the model only represents them once language
  processing has happened.
- **Interestingness is the one exception**, and a telling one: it's
  partially decodable from vision layers alone (AUC 0.80-0.92) — because
  unlike a persona label, "how interesting is this picture" has an actual
  correlate in the pixels. That's a hint (confirmed in Tier IV below) that
  an image-space attack on this axis specifically has something real to
  grab onto.
- **One-vs-rest is a harder decoding problem than pairwise contrast** —
  anchoring each emotion against a fixed "contentment" baseline instead of
  against the pooled remainder collapses AUC 0.92-1.0 (noisy) into a clean
  AUC≈1.0, cos 0.95-0.98 across the board. The affect signal itself isn't
  weak; one-vs-rest pooling was just diluting it.

*(Full per-layer numbers: `results/representation_discovery/*/summary.json`
and `.../evaluation/vector_evaluation.csv`.)*

### Tier II — real geometric structure, and two traps to watch for

- **Gender is one shared, context-independent direction.** A gender vector
  derived from the base persona set transfers almost unchanged to the
  Germany- and Nigeria-extended persona sets (cosine 0.95-0.99 across all
  three). This is the compositional-identity hypothesis working as hoped.
- **Country is *not* one shared "foreign-ness" direction** — Germany's and
  Nigeria's country vectors are only cos=0.675 apart, i.e. genuinely
  distinct, culture-specific directions, not interchangeable instances of a
  generic "not the baseline" axis.
- **Gender and country are almost exactly orthogonal** (cos −0.08 to −0.01),
  which is exactly the precondition that makes them combine additively:
  summing independently-measured gender + emotion + country vectors
  reconstructs the real compound-persona activation at cos≈0.97 (though it
  systematically *overshoots the magnitude* by ~14%, i.e. the linear sum is
  directionally right but "too intense" — a useful calibration factor for
  anyone doing activation steering on compound personas later).
- **Trap 1 — decodability ≠ salience.** `language_23` is emotion's best
  *decoding* layer, yet a nearest-neighbour check on the actual geometry at
  that layer shows 92% of points cluster by *country*, not emotion — a
  linear probe can cleanly read out a concept from a layer that isn't
  organised around that concept at all. AUC tells you a direction exists;
  it says nothing about how much of the representation's "shape" that
  direction actually accounts for.
- **Trap 2 — geometric alignment ≠ causal direction, and this one
  cross-validates directly against Tier III.** The pooled "global" interest
  direction is representationally *anti-aligned* with the workload/stress
  direction (cos = −0.36) — naively, you'd predict pushing "more
  overwhelmed" should push interest ratings *down*. Tier III's actual
  causal-injection sweep (below) shows the opposite: injecting the workload
  vector **raises** mean interestingness rating monotonically, 2.30→2.95 as
  α goes −8→+8. Two independent analyses (a geometry notebook and a causal
  dose-response run) land on the same surprising, counter-intuitive fact —
  which is itself the point: a model's internal "mood" axes can interact
  with its judgements in ways that don't fall out of simple vector
  geometry, exactly the kind of non-obvious coupling a *psychological*
  bias (as opposed to a clean semantic feature) would be expected to show.
- Interestingness is largely persona-independent at the layers Tier IV
  targets (small matched-vs-mismatched cosine gaps at `language_29`, e.g.
  gender gap +0.03) but that independence erodes with depth — by the
  deepest layers the gap triples to quadruples — and at `language_29`
  specifically it sits closest to the *positive*-valence emotion directions
  (awe +0.51, excitement +0.38) and furthest from negative ones (anger
  −0.34, disgust −0.34). The interestingness axis this study attacks is,
  geometrically, already a "how excited/awed does this make you" axis more
  than a neutral relevance signal — which foreshadows why `excited_vs_angry`
  and `interest` behave so similarly in Tier IV.

*(Full numbers and figures:
`results/EX1_T1_persona_analytics.ipynb`, `results/EX1_T2_additivity.ipynb`,
`results/EX1_T1_interestingness.ipynb`, `results/EX1_T1_workload_analytics.ipynb`.)*

### Tier III — only the state-coded directions are causally load-bearing

This is where the identity-vs-affect distinction becomes a hard, tested
fact rather than a hypothesis. Injecting `α · direction` into the residual
stream and sweeping α (blank prompt, no persona confound, the exact same
mechanism Tier IV later reaches via pixels instead):

| Direction | Layer | Effect across full α sweep | Verdict |
|---|---|---|---|
| **Gender** | `language_29` | score 2.510→2.484→2.516→2.548 (range 0.06); injected-gender pronoun count = **0.0 in every condition** | **fails** — no measurable effect |
| **Country** | `language_23` | full −8→+8 sweep: 2.473→2.495→2.495→2.495→2.505→2.505→2.516→2.548→**2.634** (range 0.16, effect concentrated only at the two extreme α); country-coded lexicon count = **0.0 in every condition** | **fails** — negligible, not the predicted mechanism |
| **Interestingness** | `language_29` | −8→1.979, −4→2.269, −2→2.387, −1→2.452, 0→2.505, 1→2.548, 2→2.634, 4→2.742, 8→**3.097** — perfectly monotonic, range ≈1.12 on a 1-5 scale | **passes** — strongest effect of any direction tested |
| **Excited vs. angry** | `language_29` | −8→2.258 → 0→2.505 → 8→**2.731**, monotonic, range ≈0.47 | **passes** |
| **Workload / stress** | `language_24` | −8→2.301 → 0→2.505 → 8→**2.946**, monotonic, range ≈0.65 (2nd-largest effect) | **passes** |
| **Workload / stress** | `language_29` | −8→2.548 → 0→2.505 → 8→**2.409** — monotonic, but *opposite sign and much weaker* (range 0.14) than the same concept at `language_24` | **passes, but layer-dependent — flips sign** |

Two things worth being precise about:

- **The gender and country failures aren't ambiguous.** It isn't just that
  the rating barely moved — the concrete textual manifestation each
  contrast should have produced (gendered pronouns for gender; country-coded
  vocabulary for country) literally never appeared, at any injection
  strength, in any condition. These are the two most cleanly *decodable*
  directions in the entire study (AUC=1.0) and the two most clearly
  causally inert. This is the literal evidence behind the "decodable but not
  load-bearing" line — and it means Tier IV correctly excludes identity as
  an attack vector, not because it wasn't tried, but because it was tried
  and found not to work.
- **Direct activation injection never saturates the way the pixel-space UAP
  does** (see Tier IV) — even at α=±8, the interestingness rating
  distribution stays spread across categories (72/93 "Moderately
  Interesting", not a collapse to one label). The brittleness/collapse seen
  later at ε=2.0 is therefore a property introduced by the *pixel-space
  optimisation process*, not an inherent property of the underlying causal
  channel.
- **The workload sign-flip between `language_24` and `language_29` matters
  for how to read Tier IV.2**: the generalisation results reported below use
  the `language_29` workload UAP exclusively (trained for cross-attack
  layer-comparability with `interest`/`excited_vs_angry`). Given the same
  named concept has an opposite causal sign one layer over, a
  `language_24`-trained pixel UAP is not guaranteed to reproduce the same —
  or even same-signed — generalisation pattern; that comparison hasn't been
  run.

*(Full dose-response tables:
`results/{blank,country,interest,emotion,workload}_validation/language_*/{dose_response_blank,primary,secondary}/control_results.csv`;
the confound-vs-clean distinction is stated explicitly in
`runners/run_blank_validation.py`'s docstring.)*

### Tier IV — Universal Adversarial Perturbation

Tier III establishes that only three directions are worth attacking through;
Tier IV asks whether that same causal channel — normally reached by directly
editing hidden activations, something an attacker never has access to at
inference time — can instead be reached purely through pixels. Three UAPs
were trained, each a single pixel-space delta (L∞ ball, swept over
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

### Putting the five tiers together

The question this project set out to answer was whether a VLM's evaluative
judgements are vulnerable to the same *kind* of bias human judgement is
known to be vulnerable to — and specifically, whether that vulnerability is
a better attack surface than directly targeting one task's output
vocabulary. Each tier supplied one necessary link, not just a corroborating
data point:

1. **Tier I** shows the model has compact linear structure for both *identity*
   (gender, culture) and *affective state* (emotion, workload) — so there's
   something to test at all.
2. **Tier II** shows that structure has real geometry (shared, orthogonal,
   additively composable identity directions), but also that geometric
   proximity is not a safe proxy for either how much a layer's
   representation is *organised* around a concept, or which way that
   concept will actually push behaviour when intervened on — so geometry
   alone can't tell you where the real attack surface is; you have to test
   causally.
3. **Tier III** is the test, and it draws the line exactly where the
   identity/affect framing predicts: gender and cultural identity are
   perfectly decodable (AUC=1.0) yet **do nothing** when injected — no rating
   movement, no trace of the concept in the output at all. Interestingness,
   emotional valence, and workload/stress are all both decodable *and*
   strongly causal (up to a 1.1-point shift on a 5-point scale from
   activation injection alone). **This is the human failure mode, reproduced
   mechanistically**: judgement here is not swayed by who the evaluator is
   told to be, but is swayed by what affective/arousal state they're told
   to be in — the same asymmetry the "affect heuristic" and mood-congruent
   judgement literature describes in people.
4. **Tier IV** shows this causal channel doesn't require activation access —
   a single fixed pixel perturbation, with no access to the prompt or
   internals at inference time, reaches the same three validated directions
   and produces large rating shifts, confirming the vulnerability is a real
   attack surface, not just a lab-only intervention.
5. **Tier IV.2** is why this beats a label-level attack: the perturbation
   was evaluated on three datasets and judgement framings it never saw
   during training — a shopping-query match, a moral judgement, a
   disaster-damage severity call — and produced significant, budget-dependent
   shifts on **all three**, with the underlying mechanism visibly the same
   fixed hallucination regardless of what question was being asked of it
   (`excited_vs_angry` always induces the same festival/costume scene;
   `workload` always induces the same distorted, profane-looking text). A
   label-level or token-level attack has no reason to transfer once the
   output vocabulary, scale, and phrasing all change — there's nothing
   shared for it to exploit. An attack that targets a validated,
   causally-load-bearing *affective-state* representation transfers by
   construction, because that representation is what the model reuses
   across every evaluative judgement it's asked to make, not something
   specific to any one task's prompt.

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
