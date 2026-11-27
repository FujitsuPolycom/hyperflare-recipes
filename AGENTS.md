# Working on Hyperflare Recipes

Hyperflare Recipes (`local-inference-lab/hyperflare-recipes`) holds the official
model recipes, reusable PyTorch and b12x model components, operator-tuning
fixtures, numerical reference implementations, qualification drivers, and model
test suites for [Flare](https://github.com/NVIDIA/hyperflare-inference)
(publicly named Hyperflare).

The core engine repository (`NVIDIA/hyperflare-inference`) owns the Helios Rust
serving runtime (`helios/`), the `flare` and `flare-cli` serving binaries, the
`flare_heliopause` extension (`heliopause/`), and the recipe-free `ignition`
preparation library and CLI runner (`ignition/`), including the 12 Helios-ABI
interface modules in `ignition/components/`. This repository depends on
`ignition[prepare]` from `NVIDIA/hyperflare-inference`, houses the model recipes
in `hyperflare_recipes/recipes/`, reusable b12x/Triton model components in
`hyperflare_recipes/components/`, operator-tuning fixtures in
`hyperflare_recipes/fixtures/`, numerical references in
`hyperflare_recipes/reference/`, and exposes the `ignition` CLI entrypoint in
`hyperflare_recipes/cli.py` by passing the official recipe catalog to
`ignition.cli.main(...)`. Qualification drivers live in `qualify/`, model and
recipe tests in `tests/`, scripts in `scripts/`, and helper tools in `tools/`.

## Required setup and commit review

On entering a clone or worktree, and after pulling changes, re-read this file,
[the design self-review](docs/design-review.md), and
[the repository prose skill](.agents/skills/repo-prose/SKILL.md). Before committing,
enable the repository's `.githooks/commit-msg` hook by following the
[hook installation instructions](docs/design-review.md#hook-installation).
Preserve any existing active hooks when installing.

Before every commit, follow the staged-diff review and acknowledgment instructions
in `docs/design-review.md`. A rejected acknowledgment requires reading that
document and reviewing the staged change before retrying; do not treat the
rejection as a formatting repair.

## Required prose review

Before writing or reviewing repository prose, read and apply
[the repository prose skill](.agents/skills/repo-prose/SKILL.md). It applies to
documentation, code comments, docstrings, names, errors, commit messages, PR titles
and descriptions, issue and review comments, plans, and technical reports. Code
tasks are included whenever they produce prose, including progress messages and
completion reports.

Every repository document has one or both of two purposes: it specifies the
architecture, or it instructs agents how to work in the repository. State
designed behavior that the code does not implement as a specification marked
designed, not implemented. State support as present qualification: what is
implemented, the hardware and configurations on which it is qualified, and what
is not qualified. Record unresolved defects, unrun checks, and pending
measurements in [docs/TODO.txt](docs/TODO.txt), not in design documents. Do not
add defect reports, investigation logs, run journals, or progress notes to the
repository.

Before committing or publishing, review the final text against the skill. Rewrite
passages that require conversation history, introduce unexplained identifiers, or
claim results beyond the stated evidence. Review edited PR descriptions and
comments before publishing them, even when no repository files change.

Every delegated task must include the skill's resolved absolute path and require
the delegate to read and follow it. Pass this requirement to further delegates,
and review returned prose before incorporating or publishing it. A review
acknowledgment does not substitute for reading the skill and reviewing the text.

## Required PR descriptions

A PR description must explain the argument for the change to a reviewer who knows
the repository but has none of the author's working context. **Every description
must answer these four questions, in this order: why, what exactly changed, what
benefit or result is claimed, and what evidence supports it.** A description that
makes the reviewer reconstruct these answers from implementation details, test
inventories, or linked PRs is incomplete. Rewrite it before publishing, even if
every individual statement is accurate.

1. **Why is this change being proposed?** Identify the affected workflow and the
   concrete defect, cost, or missing capability. Explain why it matters and why
   the chosen approach addresses it. Give the relevant dataflow, ownership, or
   execution constraint that makes the approach valid. Technical vocabulary and
   a sequence of operations do not substitute for a causal explanation. Assume
   an experienced engineer who needs local context, not introductory instruction.
2. **What exactly did this PR change?** Name the affected components and explain
   the concrete implementation and behavior changes. Make the before/after
   explicit: what is replaced, removed, or added, and how the resulting path
   works. Do not describe both behaviors in the same tense and leave the reader
   to infer the transition. Identify material numerical, compatibility, and
   resource-lifetime changes. A file list or a label such as "optimized prefill"
   does not explain what was done.
3. **What is the claimed benefit or result?** State the expected or observed
   effect precisely, distinguishing an expectation from a measured result. Give
   the headline comparison and metric direction for a measured improvement,
   with its scope and material limitations attached. Distinguish component
   measurements from whole-model or service results, and isolated changes from
   combined-stack results. Never attribute a combined result to one PR. The
   claimed benefit must be clear before presenting the supporting test and
   benchmark details.
4. **What evidence supports those claims?** Connect each material claim to the
   relevant checks or measurements: the behavior tested, baseline or reference,
   configuration, result, and limits of the conclusion. State what remains
   untested or inconclusive. Explain differences between runs when passed and
   skipped cases would otherwise appear contradictory. Keep material failures,
   skipped checks, and measurement limitations visible. Test counts and labels
   such as "qualified" cannot stand in for the tested contract and result.
   Reproduction commands must identify the required inputs and configuration
   directly or through a precise reference; placeholder arguments alone do not
   explain a measurement. Never invent evidence to complete the description.

Apply these requirements to the supporting detail:

- **Give details a clear hierarchy.** Establish the central design and principal
  correctness or compatibility risks before supporting implementation and
  validation details. Include details because they help assess the change.
  Avoid exhaustive inventories of changed files, preserved properties, buffers,
  and tests. Explain the relevance of non-obvious constraints. Keep reproduction
  instructions and merge logistics after the technical explanation.
- **Resolve references where they are needed.** Explain a component's role before
  using its internal shorthand. Replace phrases such as "ordinary tuning" or
  "canonical policy" with the relevant settings or a precise reference. Links
  may carry supporting detail, but the description must contain the central
  argument. For stacked PRs, explain what each necessary dependency supplies,
  what this diff contributes, and the required merge order; a chain of PR numbers
  alone is insufficient.
- **Remove process assurances that add no technical evidence.** Statements that
  source "matches the tested implementation," work was reviewed, or a branch is
  ready do not explain behavior or establish correctness. Report the actual
  checks and their scope. Do not add audit artifacts or development chronology.

Before publishing or updating a PR, read its description independently of the
diff and development conversation. **Do not publish it if the rationale, exact
change, claimed benefit, supporting evidence, or material limitations must be
guessed.** Repair missing explanations rather than adding headings or more raw
detail. Scale length to the change: a small fix may need only a short paragraph
and relevant validation. The content and order are mandatory; particular headings
and a fixed-length template are not.

## Contact on PRs and issues

Every PR and issue description has a Discord line after the technical
explanation, so project developers can contact the person who opens it directly.
The line gives that person's Discord username, for example
`Discord: example_user`. Write the username without a leading `@`; GitHub treats
`@name` as a mention of the GitHub account with that name. When the person
provides no Discord username, the line states that explicitly:
`Discord: no username provided`. Never omit the line. An agent opening a PR or
issue gives the username of the person it acts for, never its own identity.
When the agent does not know that username, it asks the person instead of
guessing.

## Contribution scope and rejection criteria

Hyperflare Recipes prepares supported model checkpoints on NVIDIA SM103/SM120/SM121
GPUs through FlashInfer/b12x and prepared CUDA graphs for serving by Helios in
`NVIDIA/hyperflare-inference`. Single-host execution and cross-node tensor
parallelism (`ignition export --ranks` on rank zero and `ignition join` on peer
hosts) are both in scope.

Every contribution must address an explicit maintainer request, a demonstrated
defect, a measured performance or memory problem, or inaccurate documentation.
Identify the affected workflow and relevant implementation before editing.
Hypothetical future consumers, agent-generated suggestions, and lint findings
alone do not establish a need for a change.

- **Reject unapproved scope expansion.** CPU or non-NVIDIA execution,
  interchangeable execution backends, training or fine-tuning, multi-model
  hosting and routing, unrelated service APIs, and general-purpose cluster
  orchestration require an explicit maintainer decision before implementation.
- **Reject architectural violations.** Production Python callbacks, eager
  serving, runtime compilation or recapture, steady-state GPU allocation,
  model-specific native policy or state-update algorithms, and executable
  extension mechanisms in prepared metadata violate the boundaries below.
  Flags, enums, callbacks, optional paths, and temporary implementations cannot
  be used to bypass them.
- **Reject weakened guarantees.** Preserve complete cache and speculative state,
  inactive-row behavior, asynchronous resource lifetimes, checkpoint precision,
  model semantics, and the declared model core contract. Do not remove
  validation, suppress errors, silently substitute execution paths, or relax references,
  workloads, or numerical bounds to obtain passing checks or favorable metrics.
  Numerical bounds must follow the required arithmetic and measured repeatability.
- **Reject unsupported abstractions and churn.** Do not add frameworks,
  extension points, speculative compatibility, fallback layers, or configuration
  switches without a concrete requirement. Unsolicited formatting, renaming,
  comment expansion, annotation sweeps, and file rearrangements are not
  contributions by themselves. Refactoring must directly support the requested
  behavior or remove a demonstrated source of defects or cost.
- **Require focused, meaningful validation.** Tests must exercise the affected
  behavior or contract; source-text checks and mocks that bypass the behavior
  cannot establish runtime correctness. Do not present scaffolding or partially
  wired features as supported. Identify missing hardware, inputs, and unrun
  checks explicitly. Keep each PR focused on one coherent outcome and describe
  its problem, resulting behavior, validation, and material limitations.

An explicit maintainer request covering the work is sufficient authorization
for changes within these constraints and their necessary supporting steps.
A feature request, passing tests, or a performance gain does not waive an
architectural invariant. Resolve conflicts through an explicit maintainer
decision on the design before implementing a change to that invariant; a commit
acknowledgment or routine PR approval is not that decision. Apply these criteria
to documentation and test changes that would weaken the rules as well as code.

When reviewing a contribution that violates these criteria, identify the
concrete violation and relevant rule, and recommend rejection. Do not rescue
it by adding an opt-in, weakening the rule, expanding its scope, or promising
future cleanup.

## Choose guidance by task

Read the guidance for every task affected by the work before implementing or
running it. For preparation or model validation, read the relevant model and
hardware sections of the preparation guide.

| Task | Required guidance |
|---|---|
| Install dependencies, prepare, or export an official model | [Build and run model checkpoints](#build-and-run-model-checkpoints), [model preparation and serving](docs/model-preparation.md), and [CLI commands](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/cli-design.md); [direct execution](docs/direct-execution.md) when reproducing faults with `--direct` |
| Add or change a model recipe or execution component | [Model development](docs/model-development.md), [preparation development](docs/preparation-development.md), and [core model preparation design](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/core-design.md#2-model-preparation) |
| Change tuning, compilation, memory planning, capture, or TP preparation | [Preparation development](docs/preparation-development.md), [preparation resources](docs/model-preparation.md#preparation-resources), and [compressed scale layout](docs/tp-checkpoint-layout.md) |
| Diagnose or optimize GPU performance or fixed memory | [GPU performance work](docs/performance-work.md), [model development](docs/model-development.md), and [preparation development](docs/preparation-development.md) |
| Change speculative decoding, log probabilities, or prompt scoring in recipes | [Speculative decoding design](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/mtp-design.md), [the logprob contract](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/logprobs-design.md), and [model development](docs/model-development.md) |
| Change Helios runtime, core format, Protobuf metadata, or core `ignition` | Work in [`NVIDIA/hyperflare-inference`](https://github.com/NVIDIA/hyperflare-inference) following its `AGENTS.md`, [prepared model contract](https://github.com/NVIDIA/hyperflare-inference/blob/main/helios/model-contract.md), and [execution manifest](https://github.com/NVIDIA/hyperflare-inference/blob/main/helios/manifest.md) |

For sampling or frontend metadata changes, also read
[sampling and frontend metadata](docs/model-development.md#sampling-and-frontend-metadata).
Model onboarding and changes to tokenizers, chat templates, or native prompt
encoders must satisfy the
[renderer contract and validation requirements](docs/model-development.md#prompt-renderers-and-checkpoint-boundaries).

## Build and run model checkpoints

Synchronize the environment from the repository root (the environment installs
`ignition[prepare]` and `flare-heliopause` from `NVIDIA/hyperflare-inference`,
plus FlashInfer and PyTorch):

```sh
uv sync --extra test
```

For local co-development against a checkout of `NVIDIA/hyperflare-inference`,
install its `heliopause/` extension and editable `ignition` package into `.venv`
or override `[tool.uv.sources]` locally without committing local paths.

The [preparation guide](docs/model-preparation.md) owns dependency updates,
local FlashInfer installation, GPU selection, compiler and tuning resources,
model-specific launch instructions, and hardware setup. The
[CLI design](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/cli-design.md)
defines flags, configuration, signing, and core selection.

Export runtime cores with `.venv/bin/ignition export MODEL ...` and serve them
with the native `flare` binary from `NVIDIA/hyperflare-inference`
(`flare run CORE`). For repeated exports with compatible weights, follow
[re-exporting with cached weights](docs/model-preparation.md#re-exporting-with-cached-weights).

`ignition debug-serve` is only for diagnosing model and preparation faults.
Do not use it for actual serving or apples-to-apples performance comparisons
against other runtimes, including its native, `--direct`, and `--eager` modes.
Use `ignition export` followed by `flare run CORE` for those workflows.

For debugging without export, native `ignition debug-serve` requires Heliopause's
`qualification` Cargo feature. Diagnostic `ignition debug-serve --direct` does not.

## Run tests

Run Python tests from the repository root in the synchronized environment:

```sh
.venv/bin/python -m pytest tests/cli -q
```

Tests that need a GPU skip unless `FLARE_RUN_CUDA_TESTS=1`. Tensor-parallel tests
also require `FLARE_RUN_TP_CUDA_TESTS=1` and the assigned devices in
`CUDA_VISIBLE_DEVICES`. Tests that need a checkpoint, an exported core, or a
running service read a test-specific variable, such as `FLARE_QWEN38_SNAPSHOT`
or `FLARE_API_TEST_URL`, and skip without it. Report a skipped test as an unrun
check.

Frontend tests that compare against Helios binaries require a local
`NVIDIA/hyperflare-inference` checkout. Template patch tests require
`FLARE_TEST_TEMPLATE_DIR`; see
[sampling and frontend metadata](docs/model-development.md#sampling-and-frontend-metadata).

## Architecture and ownership

Helios's shared runtime policy and separation from model computation are
non-negotiable architectural invariants across both repositories. Correct output,
performance gains, delivery pressure, and intended cleanup do not justify
violating them. Resolve architectural boundary issues before committing.

- Recipes in `hyperflare_recipes/recipes/` own model-specific composition,
  checkpoint mapping, state layouts, and graph preparation. They may retain
  canonical Transformers forwards or compose prepared components as the model
  requires.
- Put reusable model execution mechanisms in `hyperflare_recipes/components/`.
  Recipes should be able to use components independently without adopting a
  rigid model framework. Shared/routed expert overlap, projections, MoE,
  attention, MLA, GDN, Mamba, head exchange, and speculative drafting belong
  there rather than being copied between recipes.
- Core `ignition` (in `NVIDIA/hyperflare-inference`) remains completely
  recipe-free. Only the 12 Helios-ABI interface modules live in core
  `ignition/components/`. Never introduce model-name branches, recipe imports,
  or plugin discovery into core `ignition` or `helios`.
- `PreparedContract`, which Helios decodes from the core's Protobuf
  `ModelContract`, is validated data, never a runtime policy extension point.
  Never rely on per-model boolean flags (`supports_*`, `can_*`, default-false
  opt-ins) or model allowlists in Helios.
- Introducing a captured-graph operation family requires prior explicit approval
  from project owner Luke Alonso <lalonso@gmail.com>. Shape and capacity variants
  of an existing family do not constitute a new type when its operation contract
  stays unchanged. A design acknowledgment does not grant that approval.

## Operators and hardware

- Flare and Hyperflare Recipes are fully coupled to FlashInfer/b12x by design.
  There are no backend or device abstractions. Use FlashInfer/b12x and NVIDIA
  CUDA interfaces directly; do not introduce interchangeable kernel backends or
  device portability layers.
- Prefer proven b12x operators and FlashInfer sampling over duplicate kernels.
  Inspect their supported preparation, tuning, and workspace interfaces before
  introducing custom implementations.

## Python prepares; Helios serves

- Python loads weights, resolves and autotunes kernels, plans storage, warms
  execution paths, and captures supported graph variants before serving starts.
  `ignition export` writes a runtime core and closes the prepared model.
  `flare run CORE` loads that core in a separate Rust serving process.
- Helios owns sockets, request processing, tokenization, chat templates,
  scheduling, cache operations, graph replay, and output streaming. Serving
  must not call back into Python.
- CUDA graphs are the model execution boundary. Do not introduce an eager
  serving fallback, runtime compilation, graph recapture, or GPU allocation in
  steady-state execution. Reject requests outside prepared capacities explicitly.
- Keep graph executables, captured buffers, resource owners, and the CUDA
  context alive for their full native use. Model state and metadata must be
  explicit storage that Helios can manage without Python execution.
- `ignition debug-serve --direct [--eager]` is the one Python execution path, a
  diagnostic and qualification tool authorized by the project owner for
  reproducing model faults without Helios ([direct execution](docs/direct-execution.md)).
  It is never production serving or a fallback, and only its explicit flags
  select it.
- Treat KV, recurrent state, and other model histories as explicit cache state.
  Prefix reuse, branching, and speculative rollback must preserve all relevant
  state, not just token positions and KV pages.

## Model core compatibility

- Exported cores must satisfy the declared Helios core contract in
  `NVIDIA/hyperflare-inference`. Preserve checkpoint precision, declared layout
  contracts, and frontend asset contracts when updating recipes or shared
  components.

## Configuration

- Flare, Helios, recipes, and components never read an environment variable to
  choose behavior. Feature toggles, experiment switches, tuning overrides,
  debug modes, and alternate code paths selected through the environment are
  prohibited, including temporary ones.
- A setting has one of three homes: the recipe declaration, the configuration
  file, or a command-line flag, as
  [the CLI design](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/cli-design.md)
  specifies. A setting that only experiments need is an advanced flag carried
  through the plan into the recipe's `PreparationConfig`, so the plan, its
  printed report, and its recorded measurements describe what was prepared.
- `CUDA_VISIBLE_DEVICES` is the only variable Flare interprets for device
  selection; `ignition.cli.environment` sets dependency environment variables
  from configuration before importing PyTorch, b12x, Triton, or the Hugging Face
  hub.
- Tests may read variables that opt into GPU tests and locate their
  checkpoints, such as `FLARE_RUN_CUDA_TESTS`. A test variable must not change
  what the code under test does.

## CUDA kernel and model optimization authorship

Claude Opus 5.5 is not reliable for low-level kernel changes in this project.
Do not use it to design or implement low-level CUDA kernels or
changes to existing model fusion and GPU execution optimizations. This includes
CUDA C++, PTX, CuTe DSL, and Triton kernels, and changes to projection packing,
fused arithmetic, stream overlap, or scratch aliasing in optimized model paths.
Do not incorporate its generated changes to these mechanisms through another
contributor or agent.

General PyTorch modeling, recipe integration, checkpoint loading, frontend
processing, preparation orchestration, CLI/runtime work, tests, and documentation
are permitted when they do not design or change the restricted mechanisms above.

## Validation and performance

- Compare other runtimes against an exported core running through `flare run`.
  Timings from `ignition debug-serve` are diagnostic measurements and cannot
  establish comparative serving performance.
- When a potential correctness, memory, or performance deficiency appears in
  one model, inspect the shared components and equivalent paths in every
  supported recipe before scoping the fix. Track follow-up work across all
  affected models; distinguish confirmed findings from untested paths.
- Validate the affected execution path and capacities. [Model development](docs/model-development.md)
  and [preparation development](docs/preparation-development.md) define shape,
  graph-transition, state, and lifetime checks; [GPU performance work](docs/performance-work.md)
  defines measurement requirements.
- For excess fixed device memory, follow the
  [cross-model scratch workflow](docs/performance-work.md#cross-model-scratch-planning).
  Make graph-local temporaries and their dependencies visible to Inductor before
  adding manual shared pools. Keep weights, cache history, request state, and
  other values that cross graph boundaries in explicit persistent storage.
- Keep ordinary tests and raw benchmark measurements. Do not create development
  audit reports, receipts, provenance bundles, or hash inventories.

## Task tracking

Use [docs/TODO.txt](docs/TODO.txt) for miscellaneous follow-up items specific to
recipes, model components, and model qualification. Core runtime, Helios,
Heliopause, Selene, Orbit, CLI, and packaging items belong in
`NVIDIA/hyperflare-inference`'s `docs/TODO.txt`. Keep entries brief, update or
remove entries as work completes, and keep this file focused on durable guidance.
