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

## A four-tier study

| Tier | Question | Status |
|---|---|---|
| **I — Representation Discovery** (`representation/`) | Do persona/affect contrasts correspond to consistent linear directions in activation space, and where (which layer) are they strongest? | done |
| **II — Analytics** (`analytics/`) | What do those directions mean geometrically — are they shared across conditions, do they cluster, do they compose additively? | in progress |
| **III — Causal Control** (`representation/III1_vector_control/`, `control/`) | Does intervening on a direction in the residual stream (inject / scale / ablate) actually change the model's output, and in the predicted way? | in progress |
| **IV — Universal Adversarial Perturbation** (`attack/`) | Can a single pixel-space perturbation, trained against a validated direction, shift evaluation across *multiple, differently-worded* evaluation tasks — i.e. is the attack semantic rather than token-level? | in progress |

Tiers I-III are the representational groundwork: find candidate directions,
understand their geometry, and confirm which ones are actually load-bearing
for the model's judgement (as opposed to merely decodable). Tier IV is the
attack itself — it takes whichever direction Tier III validates as causally
potent and asks whether that causal effect can be induced purely through an
image perturbation, crafted once and applied universally, and whether the
resulting shift generalises beyond the task it was trained against (tested via
multiple evaluation framings — an interestingness rating task and a separate
binary relevance task).

---

## Status

This is an active, unfinished research project. The representational
groundwork (Tiers I-III) has produced working evidence that some
persona/affect directions are both decodable and causally load-bearing for
the model's evaluative output, while others (e.g. purely identity-coded
directions) are decodable but do not move the evaluation — a distinction that
directly shapes which direction Tier IV targets. The Tier IV attack itself is
under active development.

Because the study isn't complete, this README intentionally omits detailed
results, reproducibility instructions, and internal data/file formats — those
will be added once the work is further along.

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
