# Publication-Ready AI/ML Manuscript Prose Guide

This guide is for writing AI/ML manuscripts that read like finished venue submissions, not technical notes, implementation reports, or experiment logs. It covers both high-level structure and low-level prose.

The goal is always the same: make one contribution feel necessary, well-situated, precisely defined, and convincingly evaluated.

## Core Principle

A strong paper has one load-bearing claim:

> The paper identifies one necessary change in how a field should model, transform, learn, infer, or evaluate something; it explains why that change is needed; it introduces one clear idea that makes the change possible; and every major section motivates, defines, instantiates, evaluates, or calibrates that idea.

A paper can arrive at that claim through either of two entry points:

- **Prior-work gap:** existing methods make progress, but leave a specific limitation, assumption, or interface unresolved.
- **Measured observation:** a concrete behavior, redundancy, mismatch, or failure mode appears in the target system, creating an opportunity for a new transformation or learned signal.

These are not competing philosophies. A good observation usually becomes the reason the prior-work gap matters; a good gap usually needs an observation, measurement, or formal object to avoid sounding generic.

A strong layered claim still has one spine, especially in learning-guided optimization, systems-for-ML, or method-augmentation papers:

> We observe [phenomenon]; this suggests [general transformation/interface] would help if [missing object] were available; we learn/estimate/control [missing object] with [method]; experiments show [quality/cost/generalization evidence].

This is still one contribution when each layer depends on the previous one. It becomes unfocused only when the paper lists independent novelties that do not all support the same mechanism.

Everything else should move to the appendix, become a supporting diagnostic, or be removed.

Before writing, express the paper in one sentence:

> We show that [specific limitation or observed phenomenon] can be addressed by [main idea], because [mechanism], leading to [evidence].

For framework-plus-learner papers, use:

> We show that [specific inefficiency/failure] can be addressed by [formal transformation/interface] instantiated through [learned signal/module], because [mechanism], leading to [evidence across carriers/settings].

If a paragraph does not support that sentence, it probably does not belong in the main paper.

## High-Level Structure

Use one claim spine, then choose the entry point that fits the paper. The spine is:

1. Establish the field setting and why the object matters.
2. Identify the reason a change is needed: a prior-work gap, a measured observation, or both.
3. Name the load-bearing bottleneck or missing object.
4. Introduce the idea that addresses that bottleneck.
5. Instantiate the idea as a model, framework, interface, transformation, or system.
6. State any property that makes the instantiation trustworthy.
7. Evaluate the claims implied by the idea.
8. Calibrate scope, limits, and failure modes.

The two common entry routes are:

**Prior-work-gap route:**

1. Existing methods have made progress.
2. A specific limitation or assumption remains.
3. Existing responses do not fully address it.
4. The paper introduces a targeted idea.
5. The method instantiates the idea.
6. Experiments test the resulting claim.

**Observation-framework-learner route:**

1. A field setting has a recurring cost, mismatch, or failure.
2. A measured observation identifies where the waste or difficulty comes from.
3. A general transformation would remove that difficulty if an unknown object were known.
4. The paper states the formal property that makes the transformation trustworthy.
5. A learned or algorithmic module estimates the unknown object from available signals.
6. The full system integrates the transformation and module.
7. Experiments test plug-in value, competitiveness, mechanism, transfer, and limits.

Many strong papers use both routes: related work explains why the issue has not been addressed, and the observation explains why the paper's particular solution is necessary.

Do not start from implementation details, local terminology, or the experiment artifact that happened to make the project work. Start from the field's problem and vocabulary, then narrow to the paper's contribution.

## Introduction Ladder

Reference papers usually do not open with the method. They build a ladder from field setting to necessity. The ladder can be gap-led, observation-led, or a hybrid.

The common elements are:

1. **Problem importance.** State the broad problem and why it matters.
2. **Current capability.** Explain what incumbent, classical, or learning-based methods already do well.
3. **Reason for change.** State the limitation, observed behavior, mismatch, or repeated cost that motivates the paper.
4. **Prior responses.** Summarize the main families that try to handle the issue when that literature is important to the claim.
5. **Remaining bottleneck or missing object.** Identify the specific thing those families still do not solve, or the specific signal/transformation exposed by the observation.
6. **Proposed idea.** Introduce the method as a direct response to that bottleneck or missing object.
7. **Evidence.** State the main empirical finding and the diagnostics that support the explanation.

A good introduction should make the method feel inevitable. The reader should understand the necessity before seeing the method name. That necessity may come from a literature gap, a measured observation, or their combination.

Weak pattern:

> We propose Method X with components A, B, and C. We evaluate it on datasets D and E.

Gap-led pattern:

> Existing methods address [broad problem] by [strategy]. However, this strategy still depends on [assumption/interface/bottleneck]. We address this by [main idea], which [mechanism]. Experiments show [evidence].

Observation-led pattern:

> Existing systems for [broad problem] repeatedly spend effort on [measured behavior]. This suggests that [transformation/interface] would help if [missing signal] were available. We learn [signal] and use it to [system effect]. Experiments show [evidence].

## Abstract Recipe

The abstract should be a compressed version of the paper's argument, not a list of components.

Use this order, adapting the middle steps to the entry route:

1. **Paradigm or problem.** State the field setting.
2. **Reason for change.** State the limitation, measured behavior, or mismatch.
3. **Cause or opportunity.** Name the specific bottleneck, missing object, or exploitable structure.
4. **Method idea.** Introduce the proposed idea in one sentence.
5. **Differentiator.** Explain what is different from prior approaches.
6. **Evidence.** State the main empirical result and evaluation scope.
7. **Implication.** End with what the result suggests for the field.

Pattern:

> [Paradigm] has shown promise for [problem], but [limitation] remains under [setting]. This limitation arises because [cause]. We introduce [method], which [main mechanism]. Unlike [prior family], [method] [key differentiator]. Experiments on [protocol] show [result], suggesting [bounded implication].

For observation-framework-learner papers:

> [Solver/model family] is effective for [problem], but spends capacity on [repeated cost/failure mode]. We observe that [measured phenomenon] creates an opportunity to [transformation]. This transformation is useful only if [missing signal] can be identified. We introduce [method], which learns/estimates [signal] and uses it to [system effect]. Experiments with [carriers/backbones/settings] show [quality/cost result], suggesting [bounded implication].

Avoid abstract sentences that only enumerate:

> Our method has component A, component B, and component C.

Instead, state what the components accomplish together:

> The method converts [input/signal] into [useful object], then uses [feedback/constraint/model] to [effect].

## Differentiator Sentence

A venue-fit paper needs a clean sentence explaining why the contribution is not just another variant of prior work.

Pattern:

> Unlike [prior family], which [assumption/cost/interface], our method [key distinction].

Examples of useful differentiators:

- no retraining or no new labels;
- different inference-time interface;
- lower complexity;
- broader compatibility;
- different source of supervision;
- explicit handling of a failure mode prior methods leave implicit.

For framework-plus-learner papers, the differentiator may attach to either layer:

- the framework transforms a different object than prior work;
- the learned module predicts a signal prior methods assume, hand-design, or ignore;
- the interface plugs into multiple carriers rather than replacing the whole system;
- the formal property makes the transformation valid rather than merely heuristic;
- the evaluation shows the same mechanism across several backbones, scales, or variants.

Use only one or two differentiators in the main story. Too many differentiators make the paper sound unfocused.

## Venue Lexicon and Abstraction Level

AI/ML papers should describe the contribution at the level of learning, inference, adaptation, representation, and generalization. Avoid letting low-level optimization vocabulary become the main conceptual frame.

The common failure mode is mixing:

- **high-level philosophy:** broad claims about time, efficiency, search, or decision making without a precise ML object;
- **low-level implementation phrasing:** budgets, moves, swaps, 2-opt, tables, flags, heuristic steps, or wall-clock details as the central story.

Instead, write at the middle ML abstraction level:

- What object does the model see?
- What distribution or representation changes?
- What signal is learned, reused, adapted, or calibrated?
- How does test-time compute change the model's behavior?
- What is the interface between neural prediction and the external procedure?

Use classical optimization terms when they are necessary for correctness, but introduce them after the ML-facing role is clear.

Pattern:

> We use [ML-facing concept] to [role]. Concretely, this is implemented with [optimization operation], which [technical effect].

Weak:

> We run 2-opt, swap, and relocate moves under a fixed budget.

Better:

> We apply a local projection step that maps decoded transitions back to feasible, locally consistent trajectories. In our routing implementation, this projection is instantiated with candidate-limited 2-opt, swap, and relocate operators.

The second version is not hiding the operation. It names the ML-facing role first and gives the low-level operator as the implementation.

## Lexicon Translation

Use language that reflects the scientific role of the operation.

Prefer:

- `reference state` or `reference trajectory` instead of `incumbent route`, unless comparing to a solver;
- `local projection` or `feasibility restoration` before listing `2-opt`, `swap`, or `relocate`;
- `episodic memory`, `test-time prior`, or `non-parametric adaptation` instead of `memory table`;
- `logit intervention`, `logit shaping`, or `dynamic prior` instead of `score bonus`;
- `autoregressive transition` or `local decoding step` instead of `splice` or `move`;
- `inference scaling`, `test-time adaptation`, or `stateful inference` instead of `more search budget`;
- `query distribution`, `state distribution`, or `feedback granularity` instead of only `wall-clock time`;
- `compute-quality tradeoff` instead of raw `runtime` when making a conceptual claim.

Do not over-rebrand. If a method truly uses a classical operator, name it for reproducibility. The rule is ordering: first explain the ML role, then the operational instantiation.

Bad title:

> 2-opt Refinement and Memory Table

Better title:

> Local Projection and Episodic Test-Time Memory

Bad sentence:

> We keep good edges in a memory table and add their score to future actions.

Better sentence:

> We maintain a non-parametric episodic memory over successful local decisions and inject it into future logits as a gradient-free test-time prior.

## Test-Time Compute Framing

For inference-time methods, frame the paper around how extra compute changes the model's predictions, not only around how much wall-clock time is spent.

Good ML-facing questions:

- Does additional inference compute produce new information or only more samples?
- Does the method create feedback across model calls?
- Does the method adapt without weight updates?
- Does the method change the query distribution, representation, logits, or prior?
- Does the method reduce credit assignment difficulty?

Weak framing:

> Our method is faster because the budget is smaller.

Better framing:

> The method converts additional inference compute into stateful test-time adaptation: each accepted local improvement updates a dynamic prior that shapes later model calls.

Wall-clock time is still important, but it belongs in the experimental protocol and result tradeoff. The conceptual contribution should be stated in ML terms.

## Motivation Evidence Before Philosophy

Do not rely on descriptive labels alone. Terms such as `horizon mismatch`, `credit dilution`, `distribution shift`, or `constraint entanglement` become venue-fit only when tied to a measured or formal object.

For each coined diagnosis, provide at least one of:

- a compact definition;
- a diagnostic table;
- a plot;
- a toy experiment;
- a simple equation;
- a qualitative figure with a clear measured implication.

Pattern:

> We define [diagnosis] as [quantity]. Figure/Table X shows that [quantity] changes with [condition]. This motivates [method design].

Avoid long motivation prose that sounds plausible but is not grounded in a measurement or formal object.

## Fair Prior-Response Ladder

Before introducing the final gap, give prior response families their fair role.

Good flow:

1. Current methods fail under [setting].
2. One response is [family A], which helps by [mechanism], but leaves [limit].
3. Another response is [family B], which helps by [mechanism], but leaves [limit].
4. The remaining bottleneck is [specific gap].
5. This paper addresses that bottleneck with [main idea].

Do not caricature prior work. The paper sounds stronger when it explains what prior methods solve before explaining what they miss.

## Bottleneck Decomposition

A publication-ready paper usually does not say only "the task is hard." It decomposes the difficulty into one or two concrete bottlenecks.

Good bottlenecks are:

- specific enough to be testable;
- tied to known prior approaches;
- addressed directly by the method;
- reflected in the experiments.

Pattern:

> Despite progress, current methods face two obstacles. First, [bottleneck 1]. Second, [bottleneck 2]. These obstacles make [desired capability] difficult because [mechanism].

Use this only when the method truly has corresponding answers. Do not invent extra bottlenecks to make the introduction sound bigger.

## Component-to-Bottleneck Mapping

Each method component should answer a prior bottleneck.

Pattern:

> To address [bottleneck 1], we introduce [component A], which [mechanism]. To handle [remaining issue], we further introduce [component B], which [mechanism].

This prevents the method section from becoming a component list. The method should read as a chain of design choices, not a pile of modules.

Contribution lists are acceptable when the listed items form a dependency chain. A framework theorem, a learned predictor, a variant family, and a solver integration can all belong in one paper if the prose makes clear that they instantiate the same idea. The weak version is a grab bag:

> We introduce a framework, a model, several variants, a theorem, and experiments.

The stronger version explains the chain:

> The framework defines the transformation. The theorem explains why it is valid. The model predicts the missing signal required by the transformation. The variants expose a global-local tradeoff in that prediction problem. The experiments test whether the resulting plug-in improves the intended systems.

Bad:

> Our framework contains initialization, proposal generation, filtering, memory, and update.

Good:

> The proposal step exposes the model to [desired object]. The filtering step converts this proposal into [valid/evaluable object]. The memory step reuses [feedback/evidence] across later decisions. Together, these steps implement [main idea].

## Variants as Tradeoff Evidence

Variants are useful when they expose a real modeling tradeoff. They are distracting when they merely inflate the method surface.

Good variant structure:

1. Name the tradeoff before naming the variants.
2. Explain what each simpler variant captures and misses.
3. Introduce the combined or final variant as the consequence of that tradeoff.
4. Evaluate the variants on the metric that reflects the tradeoff.

Pattern:

> The target signal requires both [global/contextual property] and [local/sequential/dependency property]. A [global/parallel] variant captures [benefit] but misses [failure mode]. A [local/sequential] variant captures [benefit] but misses [failure mode]. The final variant combines them by [mechanism], and the ablation tests whether this combination improves [tradeoff metric].

Do not present variants as a menu of names:

> We propose Method-A, Method-B, and Method-C.

Present them as evidence for the design:

> Method-A tests whether global context is enough. Method-B tests whether local dependency modeling is enough. Method-C combines the two because the target signal needs both.

This is especially important for papers with AR/NAR, global/local, offline/online, teacher/student, oracle/learned, or heuristic/neural variants. The names are acceptable when the prose makes the modeling role clear first.

## Framework Before Model

When the contribution has both a general principle and a learned or engineered instantiation, separate them.

Good structure:

1. **Framework / transformation:** define the general operation, decomposition, representation, or interface.
2. **Learning / algorithmic module:** explain how the unknown part of the framework is predicted, searched, trained, or adapted.
3. **System integration:** show how the framework and module interact in the final method.

Pattern:

> The observation suggests a general framework: if [object] were known, we could [useful transformation]. The remaining challenge is to identify [object] efficiently. We therefore introduce [learned/algorithmic module] to instantiate the framework.

This structure makes the paper feel deeper because the method is not merely a model. It is a model used to realize a general problem transformation.

## Observation-Framework-Learner Papers

Some strong AI/ML papers are not best understood as "we propose a model." They are best understood as:

1. **Observation:** a measured behavior reveals avoidable cost, redundancy, instability, or mismatch.
2. **Framework:** if the right latent object were known, a general transformation would improve the system.
3. **Property:** the transformation preserves the relevant guarantee, objective order, feasibility, calibration, or interface.
4. **Learner:** a model estimates the latent object from available state, context, feedback, or labels.
5. **Integration:** the learned estimate is inserted into a solver, policy, pipeline, or inference process.
6. **Evaluation:** results show the transformed system works for the claimed reason, not only that it scores well.

This shape is common in learning-guided optimization, test-time adaptation, retrieval-augmented systems, and plug-in inference methods. The paper should make the dependency chain explicit:

> We observe [phenomenon]. If [latent object] were known, [framework] could [reduce cost/improve behavior] while preserving [property]. Because [latent object] is unavailable at test time, we learn [prediction/control signal] from [available evidence]. The resulting system [integration effect], and experiments show [mechanism-aligned evidence].

Avoid presenting the framework, theorem, architecture, and solver wrapper as separate contributions of equal weight. They are strongest when the reader sees them as a sequence: the observation motivates the transformation, the transformation exposes the missing signal, the learner supplies the signal, and the experiments validate the whole chain.

## Framework With Learned Missing Signal

A framework-plus-learner paper can be publication-ready even when it has several named parts, variants, and solver interfaces. The test is whether the parts form one dependency chain:

> One observation exposes a repeated cost or failure. One formal object defines how to exploit the observation. One learned object supplies the missing signal needed by the formal object. One experimental story tests whether the resulting system works for the claimed reason.

This pattern is especially natural in learning-guided optimization. A strong version reads like:

> We observe [stable/redundant/reusable structure]. If [latent signal] were available, [transformation] could [reduce cost/improve inference] while preserving [property]. Because [latent signal] is unavailable at test time, we learn [prediction object] from [lookahead/feedback/state/context]. The learned signal instantiates the transformation inside [solver/model/system], and experiments test [backbone lift], [competitiveness], [mechanism], and [transfer or limits].

This is not a weaker form of a model paper. The framework is the scientific object; the learner is the mechanism that makes the framework usable. The prose should make the reader remember the chain before the implementation:

- observation: what repeated structure, redundancy, instability, or mismatch was measured;
- transformation: what could be done if the right signal were known;
- property: why the transformation is trustworthy;
- learned signal: what the model predicts and how that target is obtained;
- interface: where the signal enters the existing solver, policy, or pipeline;
- evidence: which results show the interface helps for the intended reason.

For this style, a contribution list can name several pieces if each item occupies one role in the chain. The weak version lists artifacts:

> We introduce a framework, a theorem, a neural architecture, three variants, and many experiments.

The stronger version states the dependency:

> The framework defines the transformation. The theorem shows the transformation preserves the relevant property. The learner predicts the missing signal. The variants expose the global-local tradeoff in that prediction problem. The experiments test the plug-in interface across carriers and settings.

Use this pattern when a paper has the shape `observation -> formal transformation -> learned signal -> system integration`, not when it merely wraps a solver with a learned heuristic.

## Framework-Gap Bridge

When a formal framework would work with oracle information, make the missing information explicit.

Pattern:

> The framework reduces [cost/search space/complexity] when [unknown object] is available. However, obtaining [unknown object] is nontrivial because [reason]. We address this by learning/predicting/estimating [unknown object] from [available signal].

This bridge is especially useful for learning-guided optimization papers. It explains why learning is needed instead of treating the neural model as an arbitrary add-on.

## Formal Property as Trust Builder

When the method transforms the problem, solution, representation, constraints, or search space, state the key property in the main paper.

For learning-guided optimization, this is often the difference between a heuristic wrapper and a credible method. If a learned module decides what to freeze, aggregate, prune, mask, delegate, retrieve, or project, the main paper should state what remains valid after that decision.

Useful properties include:

- feasibility preservation;
- monotonicity;
- equivalence between original and transformed objectives;
- bounded complexity;
- invariance or equivariance;
- anytime behavior;
- valid recovery from a reduced representation.

The proof can live in the appendix, but the main paper should state the property and explain why it matters.

Pattern:

> This transformation preserves [property]. Therefore, improving the transformed object corresponds to [valid implication] in the original problem.

Do not bury this in the appendix if it is why the method is credible. The main paper can state a compact theorem, proposition, or property summary, then defer the proof. The prose should also explain the consequence in plain language:

> Because the reduced representation preserves [property], any improvement found after reduction can be recovered as a valid improvement for the original object.

The property should match the transformation. Feasibility preservation supports aggregation or constraint reduction; monotonicity supports objective-preserving reductions; valid recovery supports compressed or latent representations; calibration or consistency supports prediction-to-decision interfaces.

## Section Roles

Each section needs one job.

- `Abstract`: problem, gap, method idea, mechanism, main evidence. Avoid compressed implementation logs.
- `Introduction`: field context -> limitation -> existing responses -> remaining bottleneck -> method -> evidence.
- `Related Work`: organize prior work by mechanisms and assumptions, not by a list of method names.
- `Preliminaries`: define the objects needed to understand the diagnosis and method. Keep it neutral.
- `Motivation / Diagnosis`: isolate the phenomenon before presenting the method.
- `Method`: instantiate the idea and explain why each component exists.
- `Experiments`: answer the questions created by the claims.
- `Discussion`: interpret the evidence and identify when the method should or should not help.
- `Limitations`: state real boundaries of the contribution, not apologies.
- `Appendix`: hold details that support reproducibility, additional evidence, proofs, extended tables, and implementation specifics.

If a section tries to motivate, define, prove, implement, and defend at the same time, the paper starts to feel shallow because no role is performed deeply.

## Related Work

Related work should build the map that makes the contribution legible.

For each subsection, answer:

1. What family of work is this?
2. What mechanism, assumption, or interface does the family use?
3. Why is it relevant to the paper's gap?
4. What remains unresolved?

Use category-first headings when possible:

- `Direct Generalization`
- `Decomposition-Based Methods`
- `Local Policy Methods`
- `Learning-Guided Search`
- `Test-Time Adaptation`
- `Representation and Generalization`

Do not make method names the structure of the prose. Method names are allowed as anchors when a specific mechanism matters, but the paragraph should still be organized by category and limitation.

Weak:

> POMO does X. BQ does Y. LEHD does Z. INViT does W.

Better:

> One line of work improves constructive policies through stronger training objectives, architectural changes, or symmetry-aware formulations. These methods improve in-distribution construction, but their learned decision rules can still degrade under [target shift].

## Preliminaries

Preliminaries are not a second introduction and not a method preview. They should prepare notation and concepts so the next section can be precise.

Include:

- instance, solution, objective, and metric definitions;
- model interface or prediction object;
- inference/search setting that later sections modify;
- symbols used repeatedly later.

Avoid:

- strong claims;
- result interpretation;
- ablation motivation;
- implementation details;
- new terminology that is not used later;
- mathematical formality that does not support the method.

The reviewer should leave preliminaries ready to understand the diagnosis, not already pushed through the argument.

## Motivation or Diagnosis

A strong AI/ML paper often has a load-bearing phenomenon section before the method. This section makes the paper feel scientific rather than merely technical.

Good structure:

1. State the behavior, mismatch, or empirical observation.
2. Explain why common approaches induce it.
3. Define what a useful solution should preserve or change.
4. Provide compact evidence, a figure, or a diagnostic if available.
5. Transition naturally to the method.

The diagnostic does not need to prove the entire paper. Its role is to make the design problem concrete.

Use measured language:

- `can`, `often`, `suggests`, `is consistent with`, `under this protocol`;
- avoid implying all prior methods fail;
- avoid presenting the main result table as the motivation unless the table is genuinely diagnostic.

## Named Observation Section

When the paper's contribution depends on a particular failure mode, consider adding a standalone section between preliminaries and method.

The title should name the phenomenon, not the method:

- Good: `Distribution Shift in Local Subgraphs`
- Good: `Computational Bottlenecks in Large-Scale Training`
- Good: `Credit Assignment Under Sparse Feedback`
- Weak: `Why Our Method Works`
- Weak: `Motivation for Method X`

This section should do three things:

1. define the phenomenon in field terms;
2. show a compact figure, table, or conceptual diagnostic;
3. end by stating what a successful method must change or preserve.

It should not contain the full method pipeline. The method comes next as the answer to the observation.

## Method Prose

The method section should explain mechanisms, not enumerate modules.

For each component, state:

- what input it receives;
- what output it produces;
- what role it plays in the main idea;
- why it is needed;
- what it does not claim to solve.

Use equations and algorithms only after the prose has established the object being formalized. An equation should sharpen a concept, not introduce an unexplained concept.

Good sequence:

1. Short overview paragraph.
2. Component A, tied to bottleneck A.
3. Component B, tied to bottleneck B.
4. Component C, tied to the remaining feedback/efficiency/generalization issue.
5. Algorithm or formal summary.
6. Properties, complexity, or guarantees if relevant.

Avoid method sections that read like code paths.

Start with a gap-to-design bridge:

> The previous section shows that [phenomenon] limits [prior approach]. We therefore design [method] around [principle]. The method implements this principle through [component A], [component B], and [component C].

Weak:

> We initialize X, sample Y, compute Z, and update M.

Better:

> The algorithm maintains [state] so that each iteration can [purpose]. It first [mechanism], then [mechanism], and finally [mechanism]. This implements [main idea] without requiring [unwanted cost/assumption].

## Variant Triangulation

When the method has variants, do not present them as arbitrary options. Present them as a design space with complementary strengths and weaknesses.

Good pattern:

> Variant A provides [strength] but suffers from [failure mode]. Variant B provides [different strength] but suffers from [different failure mode]. The combined variant uses [mechanism] to preserve [strength A] while mitigating [failure mode B].

This is stronger than a standard ablation because it explains why the final design exists before the experiment confirms it.

Use variant names only after the tradeoff is clear. Readers should remember the design logic, not just the acronym list.

For global-local prediction methods, a useful triangulation pattern is:

> The global variant sees broad structure but may over-select or blur local distinctions. The local or autoregressive variant models dependencies precisely but can miss distributed structure or depend on initialization. The combined variant uses the global estimate to localize where to look and the local model to refine what should change.

This prose makes the combined method feel necessary before the ablation table appears.

## Experiment Protocol Block

Reference papers typically make the experimental setup easy to audit before interpreting results. Use a compact protocol block.

Recommended structure:

- `Problem Setting`: datasets, scales, distributions, benchmark sources, reference solutions.
- `Model / Solver Setting`: base model, backbone solver, training setting, or fixed model details.
- `Method-Specific Setting`: any nonstandard driver, adaptation process, inference wrapper, search budget, prompt/evolution process, or external solver interface.
- `Baselines`: categories of baselines and why they are relevant.
- `Metrics and Inference`: metrics, runtime, budgets, seeds, hardware, inference modes.
- `Main Results`: primary comparison that answers the headline claim.
- `Ablations`: component-level evidence.
- `Further Analysis`: sensitivity, transfer, robustness, failure modes, or diagnostics.

Do not define the benchmark by what has currently finished running. Define the protocol from accepted sources, then mark unavailable or pending entries in the table if necessary.

Experiments should answer claim-shaped questions:

- Does the method improve the intended target?
- Does it help for the reason claimed?
- Which components matter?
- What is the cost of the improvement?
- Where does the method fail or saturate?
- Does it transfer across intended settings?

Avoid turning the main paper into a giant leaderboard unless the contribution is itself a benchmark.

For observation-framework-learner papers, add the questions created by the chain:

- Does the observation actually occur under the target protocol?
- Does an oracle or controlled version of the framework show that the target signal is useful?
- Does the learned signal beat naive, random, handcrafted, or purely heuristic substitutes?
- Does the formal transformation preserve the property claimed in the method section?
- Does the integrated system improve because of the mechanism, not merely because it received more compute?

## Experiment Storyline

When the method plugs into or augments existing systems, structure experiments around the role it plays.

Useful order:

1. **Plug-in value:** Does the method improve the systems it is meant to augment?
2. **Field competitiveness:** Does the resulting system compare well against strong baselines?
3. **Mechanism evidence:** Why does the design work?
4. **Robustness and transfer:** Does it hold across backbones, distributions, variants, or scales?
5. **Limits:** Where does it saturate or fail?

If compatibility is part of the contribution, evaluate multiple backbones or carriers. Narrate this as evidence that the method is an interface, wrapper, augmentation layer, or general framework rather than a one-off improvement.

For plug-in optimization or inference methods, a strong experiment sequence is:

1. **Backbone lift:** show the method improves several representative carriers under the same protocol.
2. **Competitive system:** compare the best integrated system against strong classical, neural, or hybrid baselines.
3. **Naive substitute check:** compare against random selection, fixed heuristics, oracle variants, or feature-ablated versions to show the learned signal matters.
4. **Mechanism diagnostic:** measure the quantities the method claims to control, such as recall, precision, true-negative rate, reduced problem size, query distribution, or convergence behavior.
5. **Transfer and limits:** vary scale, distribution, solver, task variant, or budget, and state where the assumption weakens.

For framework-plus-learned-signal optimization papers, the experiment story can be even more explicit:

1. **Observation diagnostic:** show the measured phenomenon that creates the opportunity.
2. **Framework value:** show that exploiting the target signal would help, using an oracle, controlled, or carefully isolated comparison when possible.
3. **Learned signal quality:** evaluate whether the learned predictor captures the signal better than naive or handcrafted alternatives.
4. **Integrated plug-in value:** show lift across the intended carriers or backbones.
5. **Competitive positioning:** compare the best integrated system with strong external baselines under a clear protocol.
6. **Mechanism and limits:** include variant tradeoffs, qualitative behavior, convergence, transfer, or failure cases tied to the original observation.

This order keeps the paper from reading as a leaderboard. Early results establish that the interface is useful, middle results situate and explain the system, and later diagnostics calibrate scope.

## Results Narration

Result paragraphs should interpret the table, not merely point to it.

Good sequence:

1. State the main comparison.
2. State the strongest result.
3. Mention important exceptions in the same paragraph.
4. Explain the tradeoff or condition under which the result holds.
5. Link the result back to the paper's mechanism.

Pattern:

> Table X shows that [method] improves [metric] over [baseline family] on [setting]. The gain is largest on [condition], consistent with [mechanism]. The exception is [case], where [reason/tradeoff]. This suggests [bounded interpretation].

Do not hide exceptions. Mentioning them directly makes the positive claim more credible.

## Analysis Sections

Analysis should test the mechanism, not merely add more tables.

Strong analysis patterns:

- **Tradeoff diagnostic:** name the operational tradeoff and show where the best regime lies.
- **Oracle or upper-bound diagnostic:** test whether the target signal would help if it were available.
- **Qualitative case study:** show model or algorithm behavior on one representative instance when it reveals the mechanism.
- **Convergence or stability diagnostic:** show whether the method keeps exploring, saturates, cycles, or stabilizes.
- **Robustness diagnostic:** test distribution, backbone, budget, or variant changes tied to the claim.

For hyperparameter sweeps, do not only say which value is best. Explain the tradeoff:

> Increasing [parameter] improves [benefit] but hurts [cost/failure mode]. The best regime balances [quantity A] and [quantity B].

For oracle studies, be explicit about what is excluded from timing or cost. Oracle results support the value of a target signal; they do not represent a deployable method.

## Tables and Figures

Each table or figure should answer one question.

Good table captions:

- say what is compared;
- define the subset or protocol;
- state the metric direction;
- mention whether values are produced by this paper or cited from prior reports.

Good figures:

- show the phenomenon motivating the method;
- show the method pipeline;
- show a trend that supports the mechanism;
- show a failure mode or sensitivity.

Avoid decorative diagrams that merely restate the text.

## Citation Style

AI/ML prose should be citation-supported but not acronym-driven.

Citations should support:

- why the problem matters;
- what solver families or model families exist;
- what prior limitations are known;
- what datasets and protocols are standard;
- which specific mechanism a closely related method introduced.

Do not make broad claims in the introduction or related work without citations. Do not cite only at the end of a long paragraph when multiple separate claims need support.

Good:

> Prior work improves large-scale generalization through direct training, decomposition, local policies, and projection-based transformations \citep{...}.

Weak:

> A, B, C, and D are related methods.

## Low-Level Prose

Good AI/ML prose is direct, specific, and cumulative. Each sentence should either introduce a fact, establish a contrast, explain a mechanism, or interpret evidence.

### Sentence Roles

Use sentences with clear roles:

- **Context:** "Recent methods address [setting] by [strategy]."
- **Contrast:** "However, [limitation] remains when [condition]."
- **Mechanism:** "This occurs because [cause] changes [object]."
- **Proposal:** "We address this by [main idea]."
- **Instantiation:** "The method implements this idea through [component]."
- **Evidence:** "Experiments on [protocol] show [result]."
- **Calibration:** "These results suggest [bounded interpretation]."

Avoid sentences that only announce existence:

- "This section discusses..."
- "There are many challenges..."
- "We conduct many experiments..."
- "The results are shown in Table X..."

Replace them with the actual claim:

- "Table X shows that [method] improves [metric] while [cost/tradeoff]."

### Paragraph Shape

A strong paragraph usually has:

1. Topic sentence.
2. Mechanism or evidence.
3. Contrast or implication.
4. Transition to the next paragraph.

Avoid paragraphs that are only lists of facts. A paragraph should move the argument forward.

### Transitions

Use transitions to narrow the scope:

- `However,`
- `Consequently,`
- `To address this,`
- `In contrast,`
- `This motivates`
- `This suggests`
- `Unlike`
- `While`
- `Therefore`

Do not overuse dramatic transitions like `crucially`, `remarkably`, or `significantly`. Use them only when the result clearly deserves the emphasis.

### Wording

Prefer:

- `we study`, `we introduce`, `we evaluate`, `we find`;
- `addresses`, `mitigates`, `reduces`, `improves`, `enables`, `preserves`, `aligns`;
- `mechanism`, `interface`, `representation`, `granularity`, `search space`, `credit assignment`, `adaptation`, `generalization` when those are the actual concepts;
- `fixed`, `training-free`, `zero-shot`, `online`, or `test-time` only when technically correct;
- `suggests`, `indicates`, `is consistent with`, `under this protocol` for bounded interpretation.

Avoid:

- `currently`, `for now`, `our run`, `the code`, `this flag`, `the script`;
- `novel` repeated as a substitute for explaining the contribution;
- `first`, `pioneering`, or `state-of-the-art` without naming the exact object and evidence;
- `significant` unless statistical or practically quantified;
- `solves` unless it means exact or accepted problem-specific success;
- `simple` or `straightforward` when the mechanism is actually complex;
- defensive phrases such as `we are not claiming...` unless needed for precision;
- acronyms as sentence subjects too often;
- headings that are only acronyms plus glosses.

### Method Names and Acronyms

Use method names sparingly in prose. A named method should either be:

- the proposed method;
- a close baseline;
- a representative anchor for a mechanism;
- a citation target for a specific claim.

Layered papers may need several acronyms. This is acceptable when each acronym names a distinct role in the dependency chain:

- the transformation or formal interface;
- the learned estimator or controller;
- a modeling variant that tests a tradeoff;
- an integrated system or plug-in carrier;
- a close baseline family.

The rule is ordering. Define the mechanism first, then attach the acronym. After that, use the acronym only when it saves space or disambiguates the role.

Good:

> The decomposition requires a signal indicating which edges are likely to change. We learn this signal with [method], and compare global, local, and combined decoders to test the prediction tradeoff.

Weak:

> FOO, BAR, and BAZ are evaluated against QUX and QUUX.

Do not write a paper as a sequence of acronym comparisons. Readers should remember the mechanism before the acronym.

### Calibration

Publication-ready prose is confident but bounded.

Good:

> These results suggest that the proposed interface is a complementary axis to architecture and training.

Too strong:

> These results prove that the proposed interface solves generalization.

Good:

> The gains are largest in settings where the method's local feedback signal is most informative.

Too strong:

> The method is especially effective because this setting requires the proposed mechanism.

Good:

> We evaluate on established benchmarks and report additional diagnostics to test the proposed explanation.

Too weak:

> We ran several experiments and the results look promising.

### Novelty Claims

Novelty language should be precise enough that a reviewer can verify it.

Weak:

> We pioneer a novel framework and achieve state-of-the-art performance.

Better:

> We formalize [transformation] for [setting], prove [property], and instantiate the missing [signal/object] with [learned module]. Under [protocol], the integrated system improves [metric] over [baseline family].

For framework-plus-learner papers, the clean novelty claim is usually:

> We identify [measured phenomenon] in [setting], formalize [transformation/interface] that exploits it under [property], and learn [missing signal] needed to apply the transformation at test time. Across [carriers/settings], the resulting plug-in improves [quality/cost/generalization metric].

This lets the paper sound strong without relying on broad words such as `pioneering`. If `first` is used, attach it to the precise object:

> To our knowledge, this is the first work to learn [specific signal] for [specific transformation/interface] in [defined setting].

Use `first` only when the scope is narrow and defensible:

- first to formalize a particular transformation under a named setting;
- first to learn a specific prediction object for a specific interface;
- first to combine two modeling paradigms for a particular decision problem;
- first to show a mechanism across a defined family of backbones or distributions.

Avoid broad firstness such as `first learning-guided solver`, `first neural framework`, or `first general method` unless the related work section has earned that claim.

### Limitations

Limitations should name where the mechanism may weaken, not merely state that the method may not always work.

For learning-guided optimization, test-time adaptation, or plug-in methods, consider:

- whether the motivating observation holds outside the evaluated distribution;
- whether the learned signal depends on a specific teacher, lookahead solver, label source, or reference system;
- whether the transformation can hurt when the predicted signal has low recall, low precision, or biased errors;
- whether the method depends on scale, route length, sparsity, feedback density, or constraint structure;
- whether training cost, hardware, external solvers, or inference overhead change the practical tradeoff;
- whether some carriers, backbones, model families, or problem variants are unlikely to benefit.

Good:

> The method should help most when [phenomenon] creates redundant computation and the learned signal can identify it with sufficient recall. It may provide less benefit when [phenomenon] is weak, when constraints make aggregation brittle, or when the backbone already avoids the redundant work.

Weak:

> One limitation is that the method may not work for all tasks.

### Specificity

Replace vague claims with concrete objects.

Weak:

> The method improves performance and efficiency.

Better:

> The method reduces the optimality gap under the same inference budget and avoids the additional training cost required by [baseline family].

Weak:

> Existing approaches are not general.

Better:

> Existing approaches rely on [assumption], which can fail when [test condition] changes [input/state/search object].

### Repetition Control

Do not repeat the same proper nouns, dataset names, or scale qualifiers in every sentence. State the general problem in prose and reserve variants, scales, and table-specific qualifiers for experiments.

Pattern:

- Main prose: `routing variants`, `large-scale instances`, `constrained routing`, `benchmark instances`.
- Experiment prose: specific datasets, sizes, variants, and reference solvers.

## Appendix Discipline

The appendix should make the main paper stronger by moving detail out of the way without hiding necessary evidence.

Keep in the main paper:

- the core claim;
- the main method idea;
- enough equations or algorithms to understand the mechanism;
- the primary evidence;
- the main limitations.

For framework-plus-learner or plug-in optimization papers, also keep details that define the scientific interface:

- the learned target or signal;
- the transformation boundary;
- the recovery or validity condition;
- the point where the learned signal enters the solver, model, or pipeline;
- the minimal algorithm needed to understand how the interface changes inference.

Move to the appendix:

- full implementation details;
- hyperparameter tables;
- extended proofs;
- additional dataset descriptions;
- large per-instance tables;
- secondary ablations;
- failure cases that are useful but not central;
- reproducibility instructions.

Do not use the appendix as a dumping ground for unintegrated arguments. Every appendix section should support a claim or make the work reproducible.

The main related work can explicitly route less central categories to the appendix:

> We focus the main related work on [central families]. Additional discussion of [less central families] is provided in Appendix X.

This keeps the main paper focused while showing that the literature boundary was considered.

The appendix can extend the argument along five axes:

- **Visual intuition:** examples that make the phenomenon concrete.
- **Assumption verification:** oracle studies, sanity checks, or upper-bound diagnostics.
- **Formal support:** proofs, derivations, and property statements.
- **Detailed comparison:** close-method contrasts too long for the main related work.
- **Robustness and reproducibility:** additional distributions, seeds, hyperparameters, implementation details, and per-instance results.

Do not bury evidence required for the main claim in the appendix. Use the appendix to support, qualify, and reproduce the main paper.

## Reviewer Audit Checklist

Before submission-style polishing, audit the paper as a reviewer would.

- **Claims:** Do the abstract and introduction state exactly what the experiments and theory support?
- **Scope:** Are assumptions, settings, and limitations visible rather than hidden?
- **Claim spine:** If the paper has observation, framework, theorem, learner, and integration pieces, do they form one dependency chain?
- **Novelty:** Are `first`, `novel`, and `state-of-the-art` claims scoped to precise objects and supported by related work and experiments?
- **Reproducibility:** Are datasets, splits, reference solutions, model settings, budgets, hardware, and metrics specified enough to reproduce the main claims?
- **Baselines:** Are baseline categories justified, and are omitted baselines explained?
- **Statistics:** If results are stochastic, are seeds, variance, or repeated trials reported where they affect the claim?
- **Compute:** Is runtime or compute compared only under a clear protocol?
- **Mechanism:** Do diagnostics test the signal, transformation, or interface the method claims to improve?
- **Assets:** Are datasets, code, pretrained models, solvers, and licenses cited or documented when relevant?
- **LLM usage:** If LLMs are part of the method, data generation, evaluation, or writing policy, is the role clearly disclosed?
- **Ethics and impact:** If the work has deployment, data, fairness, privacy, or misuse implications, are they acknowledged?

## Section Checklist

Before finalizing any section, check:

- What is this section's one role?
- Which sentence states its local claim?
- Which main-paper claim does it support?
- Are all broad field claims cited?
- Are method names used only when they help?
- Is the prose category-first and mechanism-first?
- Are implementation details kept out unless necessary?
- Are terms defined before use and used consistently?
- Are claims calibrated to the actual evidence?
- Would the section still make sense to a reviewer outside the project?

## Whole-Paper Checklist

Before submission-style polishing, check:

- Can the contribution be summarized in one sentence?
- If the paper has multiple contributions, do they form a dependency chain rather than a list?
- Does the introduction create the exact questions the experiments answer?
- Does related work make the gap sharper rather than broader?
- Do preliminaries prepare the method without arguing prematurely?
- Is there a concrete bottleneck section or paragraph before the method?
- Does each method component map to a stated bottleneck?
- If the method transforms an object or problem, is the central formal property stated in the main paper?
- Do experiments test mechanism, quality, cost, and limits?
- For plug-in methods, do experiments show backbone lift, competitive comparison, naive substitute checks, diagnostics, and transfer or limits?
- Are benchmark protocols defined from accepted sources, not current availability?
- Are limitations concrete and technically honest?
- Are novelty claims precise enough to verify?
- Is the main paper free of technical-note language?
- Could a reviewer explain the contribution without mentioning code, flags, or temporary artifacts?

## Opportunity Questions

Use these questions before outlining a new paper. Each question corresponds to a conditional writing move in this guide.

- Can we summarize the paper in one sentence: problem, idea, mechanism, evidence?
- Can we build the introduction ladder from problem importance to evidence without naming the method too early?
- Can we identify one or two concrete bottlenecks instead of saying the task is generally hard?
- Can we give prior response families their fair role before stating the remaining gap?
- Can we state a clean differentiator sentence against the closest prior family?
- Can we write the abstract as paradigm -> limitation -> cause -> method -> differentiator -> evidence -> implication?
- Can we state the contribution at the ML abstraction level: learning, inference, adaptation, representation, or generalization?
- Can we translate low-level solver operations into their ML-facing role before naming the implementation?
- Can we avoid making wall-clock time, budget, moves, tables, or heuristic operators the central conceptual frame?
- Can we frame inference-time work as test-time compute, adaptation, query distribution change, logit intervention, or feedback across model calls?
- Can we ground each coined diagnosis in a measured quantity, formal definition, figure, table, toy experiment, or equation?
- Can we separate a general framework, transformation, interface, or decomposition from the learned or engineered module that instantiates it?
- Can we formulate a framework-gap bridge: if an oracle object were available, what would the framework do, and what must be learned or estimated?
- Can we state a formal property such as feasibility preservation, monotonicity, equivalence, bounded complexity, invariance, or anytime behavior?
- Can we create a named observation section before the method, titled as the phenomenon rather than the method?
- Can we support the observation with a compact diagnostic figure, table, or conceptual example?
- Can each method component be mapped to a prior bottleneck or remaining design challenge?
- Can the method section start with a gap-to-design bridge?
- Can we present method variants as a design tradeoff rather than an acronym list?
- Can we explain why the final variant combines the strengths and avoids the weaknesses of simpler variants?
- Can we structure experiments as plug-in value, field competitiveness, mechanism evidence, robustness, and limits?
- Can we evaluate multiple backbones, carriers, datasets, or settings if compatibility is part of the claim?
- Can we include an oracle or upper-bound diagnostic for the central target signal?
- Can we name the key operational tradeoff behind a sensitivity or hyperparameter study?
- Can we include a qualitative case study that reveals mechanism rather than decoration?
- Can we test convergence, stability, cycling, saturation, or exploration if the method is iterative?
- Can we route less central related work to the appendix while keeping the main related work focused?
- Can the appendix extend the argument with visual intuition, assumption verification, formal support, detailed comparison, and robustness?
- Can we make the reviewer audit pass: claims, scope, reproducibility, baselines, statistics, compute, assets, LLM usage, and ethics?
