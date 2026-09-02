Right now, Section 3 is **formalization rather than a substantial theoretical contribution**. Proposition 3.1 says that taking the Cartesian product of constraint states and conjoining admissibility predicates produces another object of the same type; Proposition 3.2 then says that if every constituent constraint is sound, repeatedly taking jointly admissible transitions satisfies every constituent constraint. Both are correct, but they follow almost immediately from the definitions. 

To turn Section 3 into real theory, I would make it prove something that is **not true merely by construction**, and that directly explains why CoRE should transfer compositionally.

1. **A semantic-equivalence/congruence theorem would be the strongest and most natural addition.** Define the behavioral semantics of a constraint as the set of feasible continuations it induces from every reachable state. Then define two constraint implementations \(\rho\) and \(\rho'\) to be semantically equivalent if they induce the same admissibility decisions and equivalent future state evolution, regardless of their names or syntactic declarations. Prove something like

$$
\rho \equiv \rho'
\quad\Longrightarrow\quad
\rho\otimes \sigma \equiv \rho'\otimes \sigma
$$

for every compatible constraint system \(\sigma\). This is a **congruence theorem**: semantic equivalence is preserved under arbitrary constraint composition. That would mathematically justify the paper's core slogan that what matters is *what a constraint does*, not what it is called. It would also provide a much stronger foundation for zero-shot transfer than the current closure proposition.

2. **Prove an expressivity theorem for the executable-constraint language.** Appendix A currently gives a particular declaration grammar with arithmetic/max-plus/min-plus updates, resets, bounds, gates, checkpoint/restore, etc. A much stronger contribution would characterize exactly what class of routing constraints this language can represent. For example, define a class of deterministic resource constraints whose feasibility depends on a finite-dimensional Markov resource state with local transition updates and bound predicates, and prove:

$$
\text{every constraint in class }\mathcal C
\text{ has an equivalent CoRE declaration.}
$$

Then give a converse showing every well-formed declaration denotes a member of \(\mathcal C\). That turns the declaration language from “here is our implementation DSL” into a formal representation result. The paper already contains the ingredients, but not this theorem.

This would also make the current limitation theoretically useful. The appendix explicitly says that cross-resource events—e.g. a driver break simultaneously resetting driving time and advancing wall-clock time—are outside the current componentwise grammar.  Instead of treating this merely as a limitation, they could characterize it precisely: **product composition is complete exactly for transition-separable constraints**, whereas coupled events require a richer joint-transition language.

3. **Characterize when simple product composition is exact.** Right now the paper assumes componentwise updates

$$
T_R(x_R,s,e)=\big(T_r(x_r,s,e)\big)_{r\in R}.
$$

A substantive theorem could say something like:

> A joint constraint system admits an exact factorization into independently executable constraint semantics iff its transition kernel is conditionally separable given the routing state/action and its admissibility predicate factors as a conjunction.

Then prove necessity and sufficiency. That would answer an important theoretical question: **when is a complex VRP genuinely just a composition of independent semantic resources, and when isn't it?**

This theorem would also give reviewers a principled interpretation of CoRE's domain of validity rather than a vague “supported semantic interface.”

4. **Prove semantic substitution invariance of the learned policy.** This is probably the best bridge from Section 3 into Section 4. Appendix B already establishes permutation symmetry and explains that the network consumes executed quantities rather than constraint identity.  More importantly, its candidate-conditioned channel is produced by actually executing the semantics, and the same coordinates are used for trained and unseen constraints. 

They could elevate this into a theorem:

> If two executable constraints produce identical candidate-conditioned semantic effects and admissibility decisions on all reachable states, replacing one with the other leaves CoRE's action distribution unchanged.

Formally, if

$$
G(x_r,s,e,\rho_r)=G(x_{r'},s,e,\rho_{r'})
$$

and

$$
A_r(x_r,s,e)=A_{r'}(x_{r'},s,e)
$$

for all reachable \(s,e\), then

$$
p_\theta(\cdot\mid z,\rho_R)
=
p_\theta(\cdot\mid z',\rho_{R\setminus r\cup r'}).
$$

That would be a genuine **representation theorem about zero-shot invariance**: a completely different declaration, name, implementation, or syntax is invisible to the learned system if its behavior is the same. This is much closer to the paper's conceptual novelty than Proposition 3.1.

5. **If they want a more ambitious theory contribution, derive a transfer bound in semantic space.** Suppose the action-energy network is \(L\)-Lipschitz in its semantic-effect inputs. Define a distance between two constraints based on the maximum difference in their normalized executed consequences:

$$
d_{\rm sem}(\rho,\rho')
=
\sup_{z,e}
\|G_\rho(z,e)-G_{\rho'}(z,e)\|.
$$

Then derive a bound such as

$$
|E_\theta(e\mid \rho)-E_\theta(e\mid\rho')|
\le L\, d_{\rm sem}(\rho,\rho'),
$$

and from the softmax obtain a corresponding bound on action-distribution change.

That would give a theoretical interpretation of *why* a new battery-like or time-like constraint might transfer: generalization depends on **semantic distance**, not task identity. This would be particularly valuable because the current unseen-constraint experiments argue empirically that the normalized consequence interface enables this behavior.

### What I would actually recommend

I would **not** try to make the paper look theoretical by adding more propositions like the current ones. Reviewers can distinguish “formal-looking” from genuinely informative theory.

The strongest feasible revision would be to restructure Section 3 around two results:

**Theorem 3.1 — Exact factorization / expressivity.** Characterize the class of constraint systems for which componentwise executable composition is exact.

**Theorem 3.2 — Semantic congruence.** If two constraints are behaviorally equivalent, they can be substituted inside any compatible composition without changing the feasible trajectory language.

Then in Section 4 add:

**Theorem 4.1 — Policy semantic invariance.** If two constraints are indistinguishable under the executed consequence interface, CoRE produces exactly the same policy under substitution.

That creates a very clean theoretical chain:

$$
\boxed{\text{constraint behavior}}
\rightarrow
\boxed{\text{semantic equivalence}}
\rightarrow
\boxed{\text{composition preserves equivalence}}
\rightarrow
\boxed{\text{CoRE is invariant to equivalent implementations}}
$$

Then the experimental story becomes much more powerful. Section 5.3 tests **composition of equivalent reusable semantic objects**, while Section 5.4 tests whether learned valuation transfers to **novel points in the same behavioral semantic space**.

That could plausibly move my assessment of the theory from about **6/10 to 8/10**, because it would make “executable constraint semantics” a mathematical object with substantive properties, rather than mainly a clean abstraction around the implementation.

Yes. The way to make this genuinely theoretical is to stop proving properties that are already built into the definitions and instead prove **local-to-global results, exact characterization results, and quantitative stability results**.

The manuscript already gives the right primitive object,

$$
\rho=(X,x^0,T,A),
$$

and composes constraints by product state, componentwise transition, and conjunctive admissibility. Its current soundness/closure propositions then follow almost immediately from that construction.  The appendix also explicitly identifies the important boundary: cross-resource state writes are not expressible by the present componentwise grammar.  And the neural model consumes executed consequences rather than resource names or declaration syntax. 

I would replace the current theoretical center with the following three results.

---

# 1. Formal setting

Let the constraint-independent routing dynamics be

$$
\mathcal B=(S,\mathcal E,C,\mathrm{Commit}),
$$

where:

* \(S\) is the routing-state space;
* \(C(s)\subseteq \mathcal E\) is the finite set of candidate routing decisions available at \(s\);
* \(\mathrm{Commit}(s,e)\) is the next routing state after decision \(e\).

An executable constraint semantics is

$$
\rho=(X,x^0,T,A),
$$

where

$$
T:X\times S\times \mathcal E\to X,
\qquad
A:X\times S\times\mathcal E\to\{0,1\}.
$$

Only \(T(x,s,e)\) for admissible transitions will matter.

Given an initial routing state \(s^0\), define recursively

$$
s^{t+1}=\mathrm{Commit}(s^t,e_t),
\qquad
x^{t+1}=T(x^t,s^t,e_t).
$$

A finite decision sequence

$$
\tau=e_0e_1\cdots e_{H-1}
$$

is feasible under \(\rho\) from \(s^0\) iff

$$
e_t\in C(s^t)
\quad\text{and}\quad
A(x^t,s^t,e_t)=1
$$

for every \(t\).

Let

$$
\mathcal L_\rho(s^0)
$$

denote the set of all such finite feasible decision sequences.

For constraints

$$
\rho_r=(X_r,x_r^0,T_r,A_r),
$$

their product is exactly the paper's construction,

$$
\bigotimes_{r=1}^m\rho_r
=
\left(
\prod_r X_r,
(x_r^0)_r,
(T_r)_r,
\bigwedge_r A_r
\right).
$$

The important question is no longer “is this product another executable semantics?” It obviously is.

The useful questions are:

1. **When can one constraint implementation be substituted for another in arbitrary formulations?**
2. **Exactly which joint constraints admit this independent product representation?**
3. **How sensitive is the learned policy to changes in executable semantics?**

Those produce nontrivial theorems.

---

# 2. Theorem 1: local semantic equivalence implies substitutability in every composition

This directly formalizes the paper's claim that the reusable object should be **constraint behavior rather than constraint identity**.

## Definition 1: interface bisimulation

Consider two constraint semantics over the same routing interface,

$$
\rho=(X,x^0,T,A),
\qquad
\tilde\rho=(\tilde X,\tilde x^0,\tilde T,\tilde A).
$$

A relation

$$
\mathcal R\subseteq X\times\tilde X
$$

is an **interface bisimulation** if:

### Initiality

$$
(x^0,\tilde x^0)\in\mathcal R.
$$

### Admissibility preservation

For every

$$
(x,\tilde x)\in\mathcal R,\quad
s\in S,\quad
e\in C(s),
$$

we have

$$
A(x,s,e)=\tilde A(\tilde x,s,e).
\tag{B1}
$$

### Successor preservation

Whenever their common admissibility value is \(1\),

$$
\bigl(
T(x,s,e),
\tilde T(\tilde x,s,e)
\bigr)\in\mathcal R.
\tag{B2}
$$

Write

$$
\rho\simeq\tilde\rho
$$

if such a relation exists.

Notice what this definition does **not** require:

* the two state spaces need not be equal;
* their coordinates need not have the same meaning;
* their implementations need not be the same;
* their syntactic declarations need not be the same;
* one implementation may even have redundant internal state.

It is a behavioral criterion.

---

## Theorem 1: contextual substitutability

Let

$$
\rho\simeq\tilde\rho.
$$

Let

$$
\Sigma=\{\sigma_1,\dots,\sigma_k\}
$$

be any finite collection of additional executable constraints over the same routing interface.

Then, for every initial routing state \(s^0\),

$$
\boxed{
\mathcal L_{\rho\otimes\Sigma}(s^0)
=
\mathcal L_{\tilde\rho\otimes\Sigma}(s^0).
}
\tag{T1}
$$

Moreover, along every common feasible sequence, the internal states of \(\rho\) and \(\tilde\rho\) remain related by \(\mathcal R\), while every constraint in \(\Sigma\) evolves identically in the two systems.

### Proof

Let

$$
\tau=e_0,\dots,e_{H-1}
$$

be arbitrary.

We prove by induction on \(t\) that, provided the prefix

$$
e_0,\dots,e_{t-1}
$$

is feasible in either composed system:

1. the routing state \(s^t\) is identical in both systems;
2. every context constraint \(\sigma_j\) has identical state in both systems;
3. the distinguished states satisfy

   $$
   (x^t,\tilde x^t)\in\mathcal R.
   $$

For \(t=0\), the routing state is \(s^0\) in both systems. Each \(\sigma_j\) starts from the same fixed initial state, and

$$
(x^0,\tilde x^0)\in\mathcal R
$$

by initiality.

Assume the claim holds at time \(t\).

For the distinguished constraint, (B1) gives

$$
A(x^t,s^t,e_t)
=
\tilde A(\tilde x^t,s^t,e_t).
\tag{1}
$$

For every context constraint \(\sigma_j\), its state, \(s^t\), and \(e_t\) are identical in the two composed systems, hence its admissibility value is identical.

Therefore

$$
A_{\rho\otimes\Sigma}(z^t,e_t)
=
\tilde A_{\tilde\rho\otimes\Sigma}(\tilde z^t,e_t).
\tag{2}
$$

Thus \(e_t\) is legal in one composition iff it is legal in the other.

Suppose it is legal.

The routing update is constraint-independent, so both produce

$$
s^{t+1}
=
\mathrm{Commit}(s^t,e_t).
$$

Each context constraint receives the same previous state, routing state, and decision, so it has the same next state in both systems.

Finally, by (B2),

$$
\left(
T(x^t,s^t,e_t),
\tilde T(\tilde x^t,s^t,e_t)
\right)\in\mathcal R.
$$

Hence

$$
(x^{t+1},\tilde x^{t+1})\in\mathcal R.
$$

The induction therefore continues.

Thus every finite prefix is feasible in \(\rho\otimes\Sigma\) iff it is feasible in \(\tilde\rho\otimes\Sigma\), proving

$$
\mathcal L_{\rho\otimes\Sigma}(s^0)
=
\mathcal L_{\tilde\rho\otimes\Sigma}(s^0).
\qquad\square
$$

---

## Why this is not tautological

We did **not** define \(\rho\simeq\tilde\rho\) as “they behave the same in every composition.”

That would indeed make the theorem vacuous.

Instead, equivalence is verified through two **one-step local conditions**, (B1)–(B2). The theorem derives from those local obligations the much stronger conclusion:

$$
\text{all horizons}
\times
\text{all routing states}
\times
\text{all possible other constraint compositions}.
$$

That is a proper compositionality theorem.

---

## Corollary 1: optimization invariance

Suppose the terminal objective depends only on the terminal routing object and not on the hidden implementation state of a constraint:

$$
f=f(s^H).
$$

Then

$$
\rho\simeq\tilde\rho
$$

implies that for every context \(\Sigma\),

$$
\boxed{
\inf_{\tau\in\mathcal L_{\rho\otimes\Sigma}}
f(\tau)
=
\inf_{\tau\in\mathcal L_{\tilde\rho\otimes\Sigma}}
f(\tau).
}
$$

So bisimilar constraint implementations define not merely the same local mask but the **same optimization problem in every formulation**.

That is a useful theoretical definition of “the same constraint.”

---

## Important application to the existing appendix

Appendix A currently says compiled fast paths are checked against interpreted declarations for equivalent states, admissibility decisions, and violations. 

With Theorem 1, the paper could turn that from an implementation sanity check into a formal methodology:

> To certify a compiled fast path for a declaration, establish an interface bisimulation between the interpreter and fast path. Theorem 3.1 then guarantees equivalence inside **every possible constraint composition**, not merely on the unit tests used for certification.

That is much stronger.

---

# 3. Theorem 2: exact characterization of when product composition is valid

This is, in my view, the most interesting theoretical result they could add.

The current paper assumes complex routing constraints factor into independently updated resource states.

But **when is that representation actually mathematically valid?**

We can answer that exactly.

---

## Joint constraint system

Suppose a complex operational requirement is initially represented as one joint executable system

$$
\eta
=
(X,x^0,T,A),
$$

and suppose its state has coordinates

$$
X
=
X_1\times\cdots\times X_m.
$$

Write

$$
x=(x_1,\dots,x_m),
$$

and let \(\pi_r\) denote projection onto coordinate \(r\).

We want to know whether there exist independent constraint semantics

$$
\rho_r=(X_r,x_r^0,T_r,A_r)
$$

such that

$$
\eta
=
\bigotimes_{r=1}^m\rho_r.
\tag{3}
$$

There are exactly two obstacles:

1. one resource's next state may depend on another resource's current state;
2. feasibility may depend jointly on several resource values.

These can be characterized precisely.

---

## Definition 2: transition locality

Coordinate \(r\) has **local transition dynamics** if for every \(s,e\),

$$
x_r=x'_r
\quad\Longrightarrow\quad
\pi_rT(x,s,e)
=
\pi_rT(x',s,e).
\tag{4}
$$

In words:

> once the routing state and decision are fixed, the next value of resource \(r\) depends only on its own current resource value.

It cannot secretly read another resource's state.

---

## Definition 3: rectangular admissibility

For each routing state \(s\) and candidate \(e\), define the set of joint resource states from which \(e\) is legal:

$$
\mathcal K_{s,e}
=
\{x\in X:A(x,s,e)=1\}.
$$

Call admissibility **rectangular** if, for every \(s,e\), there exist sets

$$
K_{r,s,e}\subseteq X_r
$$

such that

$$
\boxed{
\mathcal K_{s,e}
=
K_{1,s,e}
\times\cdots\times
K_{m,s,e}.
}
\tag{5}
$$

This means legality contains no irreducibly joint test between resource states.

---

# Theorem 2: exact factorization characterization

A joint executable constraint

$$
\eta
=
\left(
\prod_{r=1}^m X_r,
x^0,
T,A
\right)
$$

admits an exact independent factorization

$$
\eta
=
\bigotimes_{r=1}^m
(X_r,x_r^0,T_r,A_r)
\tag{6}
$$

if and only if both hold:

1. **transition locality**, Eq. (4), for every coordinate \(r\);
2. **rectangular admissibility**, Eq. (5), for every routing state \(s\) and action \(e\).

---

## Proof: necessity

Assume

$$
\eta=\bigotimes_r\rho_r.
$$

Then by product construction,

$$
T(x,s,e)
=
\bigl(
T_1(x_1,s,e),
\dots,
T_m(x_m,s,e)
\bigr).
$$

Therefore

$$
\pi_rT(x,s,e)
=
T_r(x_r,s,e).
$$

If \(x_r=x'_r\), then

$$
\pi_rT(x,s,e)
=
T_r(x_r,s,e)
=
T_r(x'_r,s,e)
=
\pi_rT(x',s,e).
$$

Thus transition locality is necessary.

For admissibility,

$$
A(x,s,e)
=
\bigwedge_{r=1}^m
A_r(x_r,s,e).
$$

Define

$$
K_{r,s,e}
=
\{u\in X_r:A_r(u,s,e)=1\}.
$$

Then

$$
A(x,s,e)=1
$$

iff

$$
x_r\in K_{r,s,e}
$$

for every \(r\). Hence

$$
\mathcal K_{s,e}
=
\prod_r K_{r,s,e}.
$$

Therefore admissibility is rectangular.

---

## Proof: sufficiency

Now suppose transition locality and rectangular admissibility hold.

For each coordinate \(r\), define a local transition function by

$$
T_r(u,s,e)
=
\pi_rT(x,s,e),
\tag{7}
$$

where \(x\) is any joint state satisfying

$$
x_r=u.
$$

Because \(X\) is the full Cartesian product, such an \(x\) exists.

Because of transition locality, if \(x\) and \(x'\) both satisfy \(x_r=x'_r=u\), then

$$
\pi_rT(x,s,e)
=
\pi_rT(x',s,e).
$$

Therefore (7) is well-defined.

From rectangular admissibility choose sets \(K_{r,s,e}\) satisfying

$$
\mathcal K_{s,e}
=
\prod_rK_{r,s,e}.
$$

Define

$$
A_r(u,s,e)
=
\mathbf 1\{u\in K_{r,s,e}\}.
\tag{8}
$$

Then for every joint state \(x\),

$$
\bigwedge_rA_r(x_r,s,e)=1
$$

iff

$$
x_r\in K_{r,s,e}
\quad\forall r,
$$

iff

$$
x\in\prod_rK_{r,s,e},
$$

iff

$$
x\in\mathcal K_{s,e},
$$

iff

$$
A(x,s,e)=1.
$$

Similarly, by construction,

$$
\left(T_r(x_r,s,e)\right)_{r=1}^m
=
T(x,s,e).
$$

Finally, let

$$
x_r^0=\pi_r(x^0).
$$

Hence the product of the constructed local semantics exactly reproduces \(x^0,T,A\).

Therefore

$$
\eta=\bigotimes_r\rho_r.
\qquad\square
$$

---

# Why Theorem 2 matters

This does something the current Section 3 does not do:

$$
\boxed{
\text{It tells us exactly when CoRE's notion of independent constraint composition is valid.}
}
$$

It also gives immediate **certificates of non-factorizability**.

---

## Corollary 2a: transition-coupling certificate

Suppose there exist

$$
x,x',s,e,r
$$

such that

$$
x_r=x'_r
$$

but

$$
\pi_rT(x,s,e)
\neq
\pi_rT(x',s,e).
$$

Then no independent product representation over the proposed resource coordinates exists.

This formally captures a cross-resource state dependency.

For example, suppose

$$
x=(t,d)
$$

contains wall-clock time and continuous-driving time, and a break is internally triggered according to \(d\).

If the next wall-clock value satisfies

$$
t^+=
\begin{cases}
t+\ell_e, & d<\tau,\\
t+\ell_e+0.75, & d\ge\tau,
\end{cases}
$$

then for two states

$$
(t,d_1),\quad(t,d_2)
$$

with the same \(t\) but different \(d\), the next time coordinate differs.

Therefore the time transition is not local in \(t\), violating Eq. (4).

That gives a theorem-level explanation of the exact limitation the current appendix describes informally for coupled break/time events. 

---

## Corollary 2b: joint-feasibility certificate

Consider two resources

$$
x_1,x_2\in[0,1]
$$

with the joint feasibility condition

$$
x_1+x_2\le1.
$$

Then

$$
\mathcal K
=
\{(x_1,x_2):x_1+x_2\le1\}
$$

is triangular, not rectangular.

For instance,

$$
(0.8,0.1)\in\mathcal K,
\qquad
(0.1,0.8)\in\mathcal K,
$$

but the coordinate recombination

$$
(0.8,0.8)\notin\mathcal K.
$$

If

$$
\mathcal K=K_1\times K_2,
$$

the first two feasible points would imply

$$
0.8\in K_1,
\qquad
0.8\in K_2,
$$

which would imply

$$
(0.8,0.8)\in K_1\times K_2,
$$

a contradiction.

So this requirement **cannot** be expressed as conjunction of independent per-resource admissibility tests.

You must instead:

* introduce a joint resource whose state contains both quantities, or
* enrich the composition grammar.

That is a meaningful theoretical boundary, not merely an implementation detail.

---

# 4. Theorem 3: quantitative stability in semantic space

The first two theorems concern **exact semantics**.

The next result can support the learning/generalization claim.

Appendix B says the neural scorer consumes executed candidate-conditioned effects such as normalized state changes, post-states, reset/trigger events, and signed admissibility margins, using a common consequence channel.  The signed margin has its sign tied to exact admissibility. 

That makes it possible to define an actual metric over constraints.

---

## Semantic observation tensor

Let

$$
U_\rho(z)
$$

denote the complete **parameter-free executable semantic observation** supplied to the learned scorer at state \(z\).

It contains, for every active resource and candidate, the quantities generated by executing the resource semantics.

Let the learned network produce the legal-action energy vector

$$
\mathbf E_\theta(U)
=
(E_\theta(e;U))_{e\in\mathcal A}.
$$

Assume that, on the domain of interest,

$$
\|\mathbf E_\theta(U)-\mathbf E_\theta(U')\|_\infty
\le
L_\theta\|U-U'\|.
\tag{9}
$$

This is an ordinary Lipschitz assumption.

If they want this to be **certified rather than assumed**, they can enforce operator-norm bounds or spectral normalization and derive \(L_\theta\) from the network layers.

---

## First: stability of the hard mask

For each constraint \(r\) and action \(e\), let

$$
y_{r,e}
$$

be the signed normalized margin used by the paper, satisfying

$$
y_{r,e}\ge0
\quad\Longleftrightarrow\quad
A_r(e)=1.
$$

Suppose two semantic systems satisfy

$$
|y_{r,e}-\tilde y_{r,e}|
\le\epsilon_m
\tag{10}
$$

for every \(r,e\), and suppose the reference state is uniformly separated from an admissibility boundary:

$$
|y_{r,e}|
\ge\gamma
\qquad
\forall r,e
\tag{11}
$$

for some

$$
\gamma>\epsilon_m.
$$

Then every margin keeps its sign.

Indeed, if \(y_{r,e}\ge\gamma\),

$$
\tilde y_{r,e}
\ge
y_{r,e}-\epsilon_m
>
0.
$$

Likewise, if

$$
y_{r,e}\le-\gamma,
$$

then

$$
\tilde y_{r,e}
<
0.
$$

Hence

$$
A_r(e)=\tilde A_r(e)
$$

for all \(r,e\), and consequently the composed legal-action sets are identical.

This is useful: it identifies the exact condition under which small semantic perturbations do **not** change feasibility.

---

# Theorem 3: semantic-policy stability

Suppose at some common routing state:

1. the two formulations have the same legal-action set \(\mathcal A\);
2. their executable semantic observations satisfy

   $$
   \|U-\tilde U\|\le\epsilon;
   $$
3. the energy map satisfies Eq. (9);
4. both use inverse temperature \(\beta\).

Let

$$
p(e)
=
\frac{e^{-\beta E_e}}
{\sum_{a\in\mathcal A}e^{-\beta E_a}},
$$

and

$$
\tilde p(e)
=
\frac{e^{-\beta\tilde E_e}}
{\sum_{a\in\mathcal A}e^{-\beta\tilde E_a}}.
$$

Then

$$
\boxed{
\|p-\tilde p\|_{\mathrm{TV}}
\le
\tanh(\beta L_\theta\epsilon).
}
\tag{T3}
$$

---

## Proof

From Lipschitz continuity,

$$
|E_e-\tilde E_e|
\le
L_\theta\epsilon
\qquad\forall e.
$$

Let

$$
a=\beta L_\theta\epsilon.
$$

Define logits

$$
\ell_e=-\beta E_e,
\qquad
\tilde\ell_e=-\beta\tilde E_e.
$$

Then

$$
|\ell_e-\tilde\ell_e|
\le a.
\tag{12}
$$

Let

$$
Z=\sum_e e^{\ell_e},
\qquad
\tilde Z=\sum_e e^{\tilde\ell_e}.
$$

From Eq. (12),

$$
e^{-a}e^{\tilde\ell_e}
\le
e^{\ell_e}
\le
e^ae^{\tilde\ell_e}.
$$

Summing gives

$$
e^{-a}\tilde Z
\le
Z
\le
e^a\tilde Z.
$$

Therefore

$$
|\log Z-\log\tilde Z|
\le a.
\tag{13}
$$

Now

$$
\log\frac{p(e)}{\tilde p(e)}
=
\ell_e-\tilde\ell_e
-
(\log Z-\log\tilde Z).
$$

Using (12) and (13),

$$
\left|
\log\frac{p(e)}{\tilde p(e)}
\right|
\le2a.
$$

Hence

$$
e^{-2a}
\le
\frac{p(e)}{\tilde p(e)}
\le
e^{2a}.
\tag{14}
$$

Set

$$
\kappa=e^{2a}.
$$

A standard likelihood-ratio argument gives

$$
\|p-\tilde p\|_{\mathrm{TV}}
\le
\frac{\kappa-1}{\kappa+1}.
$$

Since

$$
\frac{e^{2a}-1}{e^{2a}+1}
=
\tanh(a),
$$

we obtain

$$
\|p-\tilde p\|_{\mathrm{TV}}
\le
\tanh(\beta L_\theta\epsilon).
\qquad\square
$$

---

# 5. Trajectory-level consequence

We can push this beyond a one-step statement.

Suppose all matched histories of length at most \(H\) satisfy the assumptions above with the same \(\epsilon\).

Define

$$
\tau
=
\tanh(\beta L_\theta\epsilon).
$$

At each matched history,

$$
\|p_t-\tilde p_t\|_{\mathrm{TV}}
\le\tau.
$$

By maximal coupling, conditional on both policies having selected the same decisions through step \(t-1\), they can choose the same action at step \(t\) with probability at least

$$
1-\tau.
$$

Therefore there exists a coupling of the complete \(H\)-step trajectories such that

$$
P(\text{trajectories remain identical})
\ge
(1-\tau)^H.
$$

Thus

$$
P(\text{trajectories differ})
\le
1-(1-\tau)^H
\le
H\tau.
\tag{15}
$$

Suppose the terminal routing objective lies in an interval

$$
f(y)\in[f_{\min},f_{\max}]
$$

with range

$$
\Delta_f=f_{\max}-f_{\min}.
$$

If coupled trajectories do not diverge, their routing solutions and objectives are identical.

If they do diverge, their objective difference is at most \(\Delta_f\).

Therefore

$$
\boxed{
\left|
\mathbb E f(Y)
-
\mathbb E f(\tilde Y)
\right|
\le
\Delta_f
\left[
1-(1-\tau)^H
\right]
\le
H\Delta_f
\tanh(\beta L_\theta\epsilon).
}
\tag{16}
$$

This is an actual **semantic transfer bound**.

It says:

$$
\text{small perturbation in executable consequence space}
$$

implies

$$
\text{small perturbation in policy}
$$

and, over a finite horizon,

$$
\text{controlled perturbation in expected route quality}.
$$

That is far closer to a theory of semantic generalization.

---

# 6. Exact invariance becomes a useful corollary

If two constraint implementations generate exactly the same executable observations,

$$
U=\tilde U,
$$

then

$$
\epsilon=0.
$$

Theorem 3 gives

$$
\|p-\tilde p\|_{\mathrm{TV}}=0,
$$

hence

$$
p=\tilde p.
$$

Combined with Theorem 1:

> A behaviorally equivalent replacement that also preserves the model's executable observation is invisible both to the feasibility mechanism and to the learned decision rule.

This formally captures the manuscript's intended “identity-free” semantics.

The appendix currently states that the deployed model gets no resource name, registry position, syntax summary, or formulation identifier and instead operates on executed quantities.  The theorem would turn that design choice into a provable invariance.

---

# 7. What I would put in the main paper

I would rewrite Section 3 approximately as:

### 3.1 Executable constraint systems

Keep the basic

$$
(X,x^0,T,A)
$$

definition and product composition.

### 3.2 Behavioral equivalence and substitutability

Introduce interface bisimulation.

**Theorem 3.1 — Contextual substitutability.**

Bisimilar constraints may be exchanged inside arbitrary constraint compositions without changing any feasible routing trajectory or optimum.

This establishes what it formally means for “constraint behavior” to be the reusable semantic object.

### 3.3 When is constraint composition exact?

Introduce transition locality and rectangular admissibility.

**Theorem 3.2 — Exact factorization.**

A joint constraint system admits independent product decomposition iff its state transitions are coordinate-local and its action-feasible sets are rectangular in resource-state space.

Then state the two failure modes:

* cross-resource transition dependence;
* irreducibly joint admissibility.

This gives a mathematically precise scope for CoRE.

### 3.4 Feasibility preservation

The paper's current closure/soundness result can remain as a short lemma or move to the appendix. It is useful but should no longer be presented as the main theory.

Then in Section 4:

### 4.x Stability of semantic action valuation

**Theorem 4.1 — Semantic perturbation stability.**

Under a certified Lipschitz scorer and a margin-separated mask,

$$
\|p-\tilde p\|_{\rm TV}
\le
\tanh(\beta L_\theta\epsilon),
$$

with the finite-horizon objective bound above.

---

# 8. Why this would materially change my review

The theoretical story would become:

$$
\boxed{
\text{local behavioral equivalence}
\Longrightarrow
\text{substitutability under arbitrary composition}
}
$$

$$
\boxed{
\text{transition locality + rectangular feasibility}
\Longleftrightarrow
\text{exact independent constraint factorization}
}
$$

$$
\boxed{
d_{\rm semantic}\text{ small}
\Longrightarrow
d_{\rm policy}\text{ controlled}
}
$$

Those are three different kinds of result:

* an **algebraic/compositional theorem**;
* a **representation characterization theorem**;
* a **quantitative generalization/stability theorem**.

None is “the product of constraints is again a constraint,” and none assumes its conclusion in its definition.

The especially valuable result is **Theorem 2**. It makes a falsifiable statement about the structural class of VRPs that CoRE can represent. The existing appendix acknowledges that cross-resource events lie outside the current grammar; the theorem explains *exactly why*. And it tells an author how to extend the language: violations of transition locality require coupled transition factors, while violations of rectangular admissibility require genuinely joint admissibility factors.

If they can add Theorems 1 and 2 rigorously, and either prove or architecturally certify the Lipschitz premise needed for Theorem 3, I would no longer describe Section 3 as “formalization dressed as theory.” It would be a real theoretical contribution supporting the representation and transfer claims.
