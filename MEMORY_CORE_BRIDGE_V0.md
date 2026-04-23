# Memory Core Bridge Contract v0

This note captures the current design direction for a fresh memory core. It is not a retrofit specification for the existing Python memory layer. The point is to make the target ontology explicit before any Rust types, storage schema, or migration work begins.

## Intent

The core problem is not "agent memory" in the vague sense. The core problem is how to represent:

- public material
- indexical material
- derivations between them
- claims that can accumulate public standing
- provenance that remains inspectable and append-only

The system should avoid mushy stateful objects like "the agent believes X". It should instead store immutable artifacts, explicit derivations, and explicit standing assessments.

## Register Purity

There are two registers:

- `public`
- `indexical`

Register belongs to artifacts, not to the agent as a whole.

Rules:

- Every node has exactly one register.
- Register is immutable.
- Register crossing is never implicit.
- Public standing is not the same thing as private sincerity or autobiographical truth.
- Indexical origin does not disqualify a claim from later public standing, but promotion must happen through an explicit bridge event.

## Claim Kinds

The bridge logic depends on claim kind. Do not run one universal promotion rule across all claims.

Initial kinds:

- `empirical`
- `interpretive`
- `autobiographical`

Meaning:

- `empirical`: claims about publicly testable states of affairs
- `interpretive`: claims about meaning, reading, framing, or argument within a text or conceptual tradition
- `autobiographical`: claims whose primary grounding is first-person experience or personal history

Rule:

- `autobiographical` claims remain `indexical` in v0. They are not promotable to public claims.

## Allowed Register Crossings

Allowed derivation patterns:

- `public -> public`
- `indexical -> indexical`
- `public + indexical -> indexical`
- `indexical -> public` only via `publication_bridge`

Forbidden in v0:

- changing a node's register in place
- treating a personal claim as public because it is strongly felt
- promoting a public claim whose only grounding is indexical material

## Publication Bridge

`publication_bridge` is not relabeling. It creates a new public claim from prior material.

Required properties:

- it outputs a new `public` claim node
- it records the origin node ids
- it records the transformation class
- it records the operator type
- it records why the result is admissible as a public claim
- it records the public grounding that allows the claim to stand without leaning on the authority of the originating indexical stance

Suggested transformation classes:

- `abstracted`
- `generalized`
- `redacted`
- `reframed`

Suggested operator types:

- `manual`
- `pipeline`
- `llm_reviewed`

Hard rule:

- A public claim may be seeded by indexical material, but it cannot be justified solely by indexical material.

## Public Standing

Public standing is not a boolean. It is a structured status assigned to public claims.

Standing values:

- `candidate`
- `supported`
- `stable`
- `canonical`
- `revoked`

Dispute posture is separate from standing:

- `none`
- `minor`
- `live`
- `framework_split`

This separation matters. A fringe objection does not erase the standing of a highly supported public claim.

### Standing Semantics

- `candidate`: admissible for evaluation as a public claim, but not yet granted durable standing
- `supported`: enough public grounding exists to treat the claim as a legitimate public claim
- `stable`: ordinary unsupported counterclaims do not threaten the claim's status
- `canonical`: high evidential inertia; strong convergent support, challenge survival, and no live rival package of comparable force
- `revoked`: later assessment backed by materially stronger contrary evidence or framework revision has displaced the prior standing

### Inertia

"Inertia" here does not mean absolute certainty. It means that unsupported or weakly supported counterclaims do not cash out against a claim with strong public grounding.

The flat-earth example fits this:

- a person can assert flat-earth claims
- a body of sources can be assembled around the claim
- that does not grant it stable or canonical public standing if the support graph is weaker, less independent, less reproducible, and less resilient than the spherical-earth evidence package

So the system should not confuse the existence of disagreement with symmetry of standing.

## Promotion And Demotion Rules

- New public claims begin as `candidate`.
- Promotion to `supported` requires sufficient public grounding for the relevant claim kind.
- Promotion to `stable` requires that ordinary unsupported counterclaims are no longer enough to unsettle the claim.
- Promotion to `canonical` requires strong evidential inertia and no live rival package of comparable force.
- Demotion or revocation requires materially stronger contrary evidence, independent challenge, or real framework revision.
- Standing changes are append-only assessment events, not in-place edits.

## Kind-Specific Admissibility

### Empirical

Needs:

- public evidence
- independence or convergence across sources
- reproducibility or strong observational support where applicable
- explicit handling of relevant counterevidence

### Interpretive

Needs:

- anchored textual grounding
- explicit reasoning body
- awareness of viable rival interpretations
- enough support to justify the interpretation as publicly discussable, even when not uniquely final

### Autobiographical

Needs:

- first-person grounding

But in v0 these remain indexical. They do not cross into public standing.

## Append-Only Discipline

- Nodes are immutable.
- Content changes create new nodes.
- Refinements create explicit `REFINES` relationships.
- Standing changes create new assessment events.
- Nothing silently mutates from indexical to public.

## Schema v0

This is the minimal object model implied by the bridge rules.

### Node Types

#### `SourceNode`

External public material.

Examples:

- books
- papers
- scans of public texts
- webpages

Register:

- `public`

#### `NoteNode`

Authored indexical material.

Examples:

- journals
- personal glosses
- private reading notes
- reflective fragments

Register:

- `indexical`

#### `InquiryNode`

Question, task, or autonomous prompt that drives retrieval and synthesis.

Register:

- `indexical`

#### `ClaimNode`

An atomic claim or interpretation.

Properties:

- fixed register
- fixed claim kind
- concise claim text
- optional reasoning body

#### `DerivationEvent`

Immutable record of how an output node was produced.

This is the bridge object that prevents `ClaimNode` from becoming a mushy container for both content and process.

#### `StandingAssessment`

Immutable record of a public claim's standing and dispute posture at a given time.

### Core Relations

- `DerivationEvent.output -> ClaimNode`
- `DerivationEvent.inputs -> Node[]`
- `ClaimNode RESPONDS_TO InquiryNode`
- `ClaimNode REFINES ClaimNode`
- `ClaimNode CITES SourceNode`
- `ClaimNode CHALLENGES ClaimNode`

Suggested input roles for derivation inputs:

- `seed`
- `grounding`
- `rival`
- `question`
- `prior_version`

### Minimal Storage Shape

#### `nodes`

- `node_id`
- `node_type`
- `register`
- `created_at`
- `content_hash`

#### `claims`

- `node_id`
- `claim_kind`
- `claim_text`
- `reasoning_body`

#### `derivation_events`

- `event_id`
- `event_type`
- `output_node_id`
- `operator_type`
- `runtime_fingerprint`
- `created_at`
- `bridge_rationale`

#### `derivation_inputs`

- `event_id`
- `input_node_id`
- `input_role`

#### `citations`

- `claim_node_id`
- `source_node_id`
- `locator_type`
- `locator_value`
- `quote_hash`

#### `standing_assessments`

- `assessment_id`
- `claim_node_id`
- `standing`
- `dispute_posture`
- `assessor_type`
- `rationale`
- `supersedes_assessment_id`
- `created_at`

#### `standing_assessment_inputs`

- `assessment_id`
- `input_node_id`
- `input_role`

Suggested standing assessment input roles:

- `support`
- `challenge`
- `rebuttal`

#### `semantic_relations`

- `from_node_id`
- `to_node_id`
- `relation_type`

## Hard Invariants

- `NoteNode.register = indexical`
- `InquiryNode.register = indexical`
- `ClaimNode.register` is immutable
- `StandingAssessment` may target only `public` claims
- `REFINES` cannot cross registers
- `publication_bridge` must output a new public `ClaimNode`
- `publication_bridge` must include at least one public grounding input
- a public claim cannot be grounded only by indexical inputs
- citations that justify a public claim must resolve to public sources
- all state evolution is append-only

## Why This Probably Belongs In A Fresh Repo

This spec is intentionally not shaped around the existing Python memory layer.

Reasons:

- the current Python memory path is mostly session-state plus promoted snippets
- this design wants a first-class ontology for claims, derivations, and standing assessments
- trying to retrofit that ontology into the current memory tables risks dragging old assumptions forward
- a clean Rust core can enforce invariants earlier and more honestly

That does not mean the current repo is useless. It remains a useful reference implementation for:

- corpus ingestion
- chunking and retrieval experiments
- orchestration patterns
- test fixtures that can later be ported or mirrored

But if the goal is a real memory kernel rather than incremental patching, a fresh repo is likely the cleaner move.

## Next Step

Before writing Rust types or SQLite DDL, pressure-test this contract:

- Are the claim kinds sufficient?
- Should autobiographical claims always stay indexical?
- Does `canonical` need to be split into multiple public-standing tiers?
- Are there bridge cases for interpretive claims that need a narrower admissibility rule?

If the answers are stable, then the next artifact should be Rust enums plus SQLite DDL derived directly from this note.