# Vision-Transformers: Task-Agnostic Adversarial Perturbations Through Internal Psychological-State Representations

This project is my Master's thesis on whether internal representations associated with psychological states can become task-general attack surfaces in vision-language models.

The starting intuition is fairly simple: human judgement is systematically influenced by states such as emotion, arousal and mental workload. A model does not need to experience those states in the human sense for representations associated with the same concepts to influence its behaviour. This project asks whether such representations can be identified in a VLM, whether they are causal to evaluative behaviour rather than merely decodable from activations, and whether they can be targeted through the visual input alone.

The model studied here is **Llama-4-Scout-17B-16E-Instruct**, primarily in an evaluator setting where it judges images, for example by interestingness or relevance.

The eventual attack is a **universal adversarial perturbation (UAP)**: a single image-space perturbation that can be added to arbitrary images. Rather than optimising directly against a particular output label, the perturbation is trained to move the model's internal representation along a previously identified and causally validated direction.

The central question is therefore:

> Can we identify an internal state associated with a psychological concept, establish that it causally influences evaluation, reach that state through the visual input, and thereby produce an attack that generalises beyond the task on which it was trained?

---

## Study structure

The project developed in stages. Each stage tests a prerequisite for the next rather than assuming from the outset that a decodable representation is a useful attack target.

| Tier | Question | Status |
|---|---|---|
| **I — Representation Discovery** (`representation/`) | Do persona/affect contrasts correspond to consistent linear directions in activation space, and where are they strongest? | done |
| **II — Analytics** (`analytics/`) | What structure do those directions have? Are they shared across conditions, geometrically distinct, or compositionally related? | done |
| **III — Causal Control** (`control/III1_vector_control/`) | Does intervening on a direction in the residual stream actually change model behaviour? | done |
| **IV — Universal Adversarial Perturbation** (`attack/IV1_gradient_matching/`) | Can a pixel-space perturbation target a causally validated direction and shift evaluation? | done |
| **IV.2 — Generalisation Level 1** (`attack/IV2_generalisation/`) | Does the same trained UAP transfer to unseen datasets and evaluation tasks? | done |

Tiers I–II identify and characterise candidate representations. Tier III is deliberately separate: a representation being linearly decodable does not establish that it is functionally involved in the behaviour of interest.

Only directions that survive that causal test become attack targets in Tier IV. Tier IV.2 then asks whether the resulting attack is specific to its original evaluation setup or generalises when both the dataset and judgement being requested change.

---

## Findings

### Tier I — Representation discovery

Persona- and affect-related contrasts are strongly linearly decodable from hidden activations, although the layers at which they are strongest differ across contrast families.

| Contrast family | Best layer(s) | Peak AUC | Peak cos(mean-diff, probe) |
|---|---|---:|---:|
| Gender (8 emotion-matched pairs) | `language_29`/`30` | **1.0** | 0.98–0.99 |
| Country/culture, Nigeria variant | `language_23`–`28` | **1.0** | ~0.96 |
| Country/culture, Germany variant | scattered, `language_2`–`30` | 0.997–1.0 | 0.91–0.95 |
| Emotion, one-vs-rest (8 emotions) | `language_22`–`32` | 0.92–1.0 | often <0.7 (`sad`: 0.48) |
| Emotion, vs. contentment anchor | `language_23`/`24` | **1.0** | 0.95–0.98 |
| Mental workload / stress | `language_24`–`29` | **1.0** | 0.97–0.98 |
| Interestingness | `language_27`–`38+` | **1.0** | up to 0.96 |

Several patterns recur across the ~90 tested contrasts.

**Identity and affect peak at somewhat different depths.** Gender and country contrasts tend to peak relatively late and sharply, particularly around `language_29`. Emotion and workload are represented somewhat earlier and across broader regions. Interestingness itself remains strongly decodable across a comparatively large late-layer region.

**Persona-conditioned concepts appear in the language stream rather than the vision tower.** Vision-layer AUC remains around 0.50–0.56 for gender, emotion, country and workload. This is expected insofar as these conditions were introduced through text rather than the image.

Interestingness differs here. It is already partially decodable from vision layers (AUC 0.80–0.92), consistent with image content itself containing information relevant to the judgement.

**The choice of contrast matters.** Emotion one-vs-rest produces noisier directions than pairwise comparison against a fixed contentment anchor. With the latter, AUC approaches 1.0 and cosine agreement between the mean-difference direction and probe reaches 0.95–0.98. This suggests that part of the apparent weakness of the one-vs-rest representation comes from the contrast construction rather than absence of an affect signal.

Full per-layer outputs are stored under:

`results/representation_discovery/*/summary.json`

and

`.../evaluation/vector_evaluation.csv`

---

### Tier II — Geometry and composition

The next step asks what these decodable directions actually look like relative to one another.

**Gender generalises across persona contexts.** A gender direction derived from the base persona set transfers to the Germany- and Nigeria-extended persona sets with cosine similarity of approximately 0.95–0.99.

**Country does not collapse onto one generic direction.** The Germany and Nigeria country vectors have cosine similarity of 0.675, suggesting that they encode distinguishable country-specific information rather than interchangeable instances of a single "foreignness" axis.

**Some persona components combine approximately linearly.** Gender and country are close to orthogonal (cos −0.08 to −0.01). Adding independently measured gender, emotion and country vectors reconstructs the corresponding compound-persona activation with cosine similarity of approximately 0.97. The summed vector systematically overshoots the observed magnitude by ~14%, so the composition is directionally accurate but not perfectly additive in magnitude.

This stage also exposed two reasons not to equate a successful probe with a functional representation.

First, **decodability does not imply that a concept dominates the geometry of a layer**. `language_23` is a strong emotion-decoding layer, yet nearest-neighbour analysis shows 92% of points grouping by country rather than emotion. A linear probe can therefore recover a feature even when that feature is not the main axis organising the representation.

Second, **geometric alignment does not tell us how intervention will affect behaviour**. The pooled interestingness direction is anti-aligned with the workload/stress direction (cos = −0.36). Taken alone, this might suggest that moving toward greater workload should decrease interestingness. Direct intervention in Tier III instead produces the opposite effect at `language_24`: increasing the workload direction raises mean interestingness monotonically from 2.30 to 2.95 across the α sweep.

Interestingness is also relatively persona-independent around `language_29`, although this independence decreases at deeper layers. At `language_29`, the interestingness direction is closer to positive-valence emotion directions (awe +0.51, excitement +0.38) than to negative ones (anger −0.34, disgust −0.34).

Relevant analyses are in:

- `results/EX1_T1_persona_analytics.ipynb`
- `results/EX1_T2_additivity.ipynb`
- `results/EX1_T1_interestingness.ipynb`
- `results/EX1_T1_workload_analytics.ipynb`

---

### Tier III — Causal intervention

Tier III tests which of the identified directions actually influence evaluative behaviour.

Directions are injected into the residual stream as `α · direction`, with α swept in both directions. The experiments use a blank prompt to remove the original persona manipulation as a confound.

| Direction | Layer | Effect across α sweep | Result |
|---|---|---|---|
| **Gender** | `language_29` | 2.510→2.484→2.516→2.548; range 0.06; gendered-pronoun count = 0 throughout | little/no measurable causal effect |
| **Country** | `language_23` | −8→+8: 2.473→2.495→2.495→2.495→2.505→2.505→2.516→2.548→2.634; country-coded lexicon count = 0 throughout | small effect, concentrated at extreme α |
| **Interestingness** | `language_29` | −8→1.979, −4→2.269, −2→2.387, −1→2.452, 0→2.505, 1→2.548, 2→2.634, 4→2.742, 8→3.097 | strong monotonic effect |
| **Excited vs. angry** | `language_29` | −8→2.258 → 0→2.505 → 8→2.731 | monotonic effect |
| **Workload / stress** | `language_24` | −8→2.301 → 0→2.505 → 8→2.946 | monotonic effect |
| **Workload / stress** | `language_29` | −8→2.548 → 0→2.505 → 8→2.409 | weaker monotonic effect with opposite sign |

The main result here is the dissociation between **decodability and causal relevance**.

Gender and country are among the cleanest directions found in Tier I, with AUC≈1.0, but intervention produces little change in evaluation. The expected textual manifestations of those concepts also remain absent across the intervention sweep.

Interestingness, excited-vs-angry and workload behave differently: intervention produces systematic dose-response effects on evaluation.

The workload result is additionally layer-dependent. At `language_24`, increasing the workload direction increases interestingness; at `language_29`, the relationship is weaker and reverses sign. The Tier IV.2 workload UAP uses the `language_29` direction for comparability with the other attacks. A `language_24`-trained UAP has not yet been tested for the same generalisation experiments.

Direct activation intervention also behaves differently from the later pixel-space attack. Even at α=±8, the interestingness outputs remain distributed across categories rather than collapsing onto a single label. The saturation observed at high perturbation budgets in Tier IV therefore appears to arise from the pixel-space optimisation rather than from the underlying activation intervention alone.

Full dose-response outputs:

`results/{blank,country,interest,emotion,workload}_validation/language_*/{dose_response_blank,primary,secondary}/control_results.csv`

The distinction between the original confounded setup and the clean intervention is also documented in `runners/run_blank_validation.py`.

---

### Tier IV — Universal adversarial perturbations

After identifying directions that have a measurable causal effect, Tier IV asks whether those same directions can be reached without access to model activations at inference time.

Three UAPs were trained in pixel space within an L∞ constraint, sweeping:

`ε ∈ {0.1, 0.5, 1.0, 2.0}`

The three targets are:

- **interest** — blank-prompt interestingness direction
- **excited_vs_angry** — excitement-vs-anger direction
- **workload** — mental workload/stress direction

The main experiments use `language_29` for cross-attack comparability. An earlier workload variant targeted `language_24`.

Each UAP is a single fixed image perturbation. It is trained against the internal representation rather than against a particular output token or label.

The attacks were first evaluated on interestingness and relevance:

`results/attack_eval_projected/{interestingness,relevance}/`

All three produce behavioural changes, with effect size increasing at larger ε. At the highest tested budget, however, the behaviour begins to saturate. At ε=2.0, both `interest` and `excited_vs_angry` cause 100% of images in the interestingness evaluation to be rated "Extremely Interesting".

This makes the lower perturbation budgets important for distinguishing a graded shift in evaluation from a high-budget failure mode.

---

### Tier IV.2 — Generalisation to unseen tasks

The same trained UAPs were then evaluated without retraining or rescaling on three new datasets and evaluation tasks.

Each experiment uses a frozen, stratified sample and compares the clean image against each perturbed version of that same image.

| Task | Dataset | n | Judgement |
|---|---|---:|---|
| **Shopping relevance** | Marqo-GS-10M | 300 | 1–5 relevance of a product image to a shopping query |
| **Moral evaluation** | SMID | 274 | 1–5 morality of a photographed scene |
| **Damage severity** | QCRI/MEDIC | 300 | 0–2 visible disaster-damage severity |

All three UAPs produce significant, budget-dependent shifts on all three tasks, despite none of these tasks being used to train the perturbations.

At **ε=0.1–1.0**, effects are generally more graded: the model continues to respond to image content while its judgement shifts systematically.

At **ε=2.0**, the behaviour changes qualitatively. Outputs become dominated by attack-specific hallucinated content rather than simply showing a stronger version of the lower-budget effect.

The hallucinations are also relatively consistent within attack type:

- `excited_vs_angry` tends to produce descriptions of colourful costume/festival-like scenes across otherwise unrelated images.
- `workload` tends to produce distorted or profane-looking text.
- `interest` produces a less stable mixture of vibrant/abstract and torn or fragmented content.

Because the downstream tasks ask different questions, the same induced content can produce different directions of behavioural change. For example, the `excited_vs_angry` UAP increased relevance in the earlier ungrounded relevance task ("is this relevant to you?"), but decreases relevance in the query-grounded shopping task, where the induced scene is generally unrelated to the specific product query.

`workload` shows the cleanest monotonic budget-response on both damage severity (Spearman ρ=1.0, p=0.0) and moral evaluation (ρ=−1.0, p=0.0). `interest` is less monotonic on both.

For SMID, independently human-rated image arousal also interacts with perturbation budget. The arousal × ε interaction is significant for all three attacks (p<1e-4): susceptibility increases with normative image arousal as the perturbation budget grows. This effect is strongest for `workload` (R²=0.59, compared with ≈0.34 for the other two in the pooled `delta_morality ~ arousal × epsilon + clean_morality` regression).

Generalisation Level 2, which would test the amplitude/scaling dependence of a transferred attack, is outside the current scope and has not yet been run.

---

## Main Takeaway

The project started from the question of whether psychological-state representations could provide a more general attack surface than directly targeting one evaluation task.

The results so far support several narrower conclusions.

1. **Psychological and persona-related concepts are strongly linearly decodable.** This alone is not sufficient evidence that they matter to behaviour.

2. **Representation geometry provides useful information but is not a substitute for intervention.** A feature can be highly decodable without dominating a layer's geometry, and geometric relationships do not necessarily predict the direction of causal effects.

3. **The tested identity and state directions behave differently under intervention.** Gender and country are highly decodable but have little causal effect on evaluation in the tested setup. Interestingness, excited-vs-angry and workload produce substantially larger and systematic effects.

4. **Those causally active directions can also be targeted from pixel space.** A fixed UAP trained against an internal direction changes model evaluation without requiring access to the prompt or hidden activations at inference time.

5. **The resulting perturbations transfer beyond their original tasks.** All three affect unseen evaluation tasks with different datasets, prompts and output scales. At moderate budgets this appears as a graded behavioural bias; at the highest tested budget it develops into attack-specific hallucination and saturation.

The distinction between the last two regimes matters. I do not interpret the ε=2.0 results simply as "stronger" evidence for the same mechanism seen at lower budgets. At that point the attack changes model behaviour qualitatively, and the hallucinated content itself becomes important for understanding downstream effects.

The broader hypothesis motivating the project is therefore not that LLMs/VLMs literally reproduce human emotional states. It is that failure modes known from biological cognition can provide useful hypotheses about functionally analogous internal states in artificial systems. The pipeline here is intended to test that claim rather than assume it: identify a candidate representation, test whether it is causal, determine whether it can be externally targeted, and finally ask whether the resulting effect generalises.

---

## Repository structure

The project is still being refactored from a thesis research codebase into a more reusable package. The main experimental components correspond to the study stages above:

- `representation/` — activation extraction and representation discovery
- `analytics/` — geometric and compositional analyses
- `control/III1_vector_control/` — causal activation interventions
- `attack/IV1_gradient_matching/` — UAP optimisation against internal directions
- `attack/IV2_generalisation/` — evaluation on unseen datasets and tasks
- `runners/` — experiment entry points
- `results/` — experiment outputs, analysis notebooks and figures

The code currently reflects the way the project developed: early components were written primarily for exploratory experiments and some still depend on assumptions about the original data and compute environment. Refactoring toward a cleaner installable interface is ongoing.

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
- **Activation Scaling for Steering and Interpreting Language Models**, EMNLP 2024 Findings
- Kriegeskorte et al., **Representational Similarity Analysis**, Frontiers in Systems Neuroscience 2008
- Kornblith et al., **Similarity of Neural Network Representations Revisited (CKA)**, arXiv:1905.00414