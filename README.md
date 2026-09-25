# Mat-Sim Agent

**An autonomous AI agent that runs complete materials-simulation workflows on HPC, evaluates its own results against the literature, and reports honestly — including when a result is imperfect.**

Mat-Sim Agent turns a plain-language question into a full simulation study. It picks the right physics engine for each material, generates the input files, runs the calculations on GPU, compares every computed value against curated references, and flags deviations with an explanation instead of hiding them.

It handles **both metals and polymers** — two domains that normally require completely different tools:

| Domain | Engine | Method |
|---|---|---|
| **Metals** | LAMMPS | interatomic potentials (EAM), lattice & elastic constants |
| **Polymers** | Amber / AmberTools | force fields (GAFF2), molecular dynamics |

---

## ⚠️ Important — the code is published as-is (developed on the Drago HPC cluster at CSIC)

This code is shared **exactly as it ran** on its original environment. It has **not** been generalised, and it contains **hard-coded, environment-specific paths** — and, in one place, a **personal account path**. Review and edit these before running the code anywhere else.

**Contains a personal account path (edit this first):**

| File | Line | What it is |
|---|---|---|
| `run_yaml_experiment.py` | line 1 | shebang pointing to a personal conda environment (`.../aadib/.conda/...`) — replace with `#!/usr/bin/env python3` or your own interpreter |

**Contains cluster installation paths (adapt to your system):**

| File | What to check |
|---|---|
| `amber_functions.py` | `AMBER_BIN` / `AMBERHOME` — absolute paths to the Amber/AmberTools install |
| `polymer_builder.py` | Amber paths and the `packmol` binary location |
| `lammps_functions.py` | LAMMPS `site-packages` paths added to `sys.path` |
| `atomagents_langgraph_mixed.py` | the `potential_repository/` location for interatomic potentials |

> These paths are specific to the machine this was developed on. On another system, point them to your own installations (or replace them with environment variables such as `$AMBERHOME`, or load the corresponding HPC modules). The code is otherwise unmodified.

If you'd like help adapting it to a particular setup, get in touch — it can be reviewed case by case.

---

## Key ideas

- **Zero simulation code for the user.** You describe *what* to compare (a YAML file); the agent writes the LAMMPS/Amber input files, builds molecular topologies, and orchestrates the run.
- **Provenance, not black box.** Every value in the output records which engine produced it (`LAMMPS`, `LAMMPS_derived`, `Amber`). Reference values are used only for comparison — never mixed into the results as if computed.
- **Honest evaluation.** Each computed property is compared to curated literature. When it deviates, the agent flags it and explains the likely physical cause rather than presenting an unreliable number as correct.
- **Fully open, local stack.** A locally-hosted open-source LLM (e.g. Qwen2.5 via Ollama) orchestrates open simulation tools. No data leaves the infrastructure; no per-call API costs.

---

## Architecture

```
                    Scientific question (natural language + YAML)
                                     │
                                     ▼
                          ┌──────────────────────┐
        Knowledge base ──▶│   LLM orchestrator    │   plans · routes · decides
        (RAG + web)       │  (local open model)   │
                          └──────────┬───────────┘
                        metal        │        polymer
                 ┌───────────────────┴───────────────────┐
                 ▼                                         ▼
        ┌─────────────────┐                       ┌──────────────────┐
        │ LAMMPS · metals │                       │ Amber · polymers │
        │ lattice const.  │                       │ topology (SMILES)│
        │ elastic → E     │                       │ density (MD)     │
        │ density         │                       │ Young modulus MD │
        │ surface energy  │                       │ glass transition │
        │ fracture tough. │                       └──────────────────┘
        └─────────────────┘
                 └───────────────────┬───────────────────┘
                                     ▼
                       Evaluation vs literature
                    (flags deviations, explains cause)
                                     │
                                     ▼
                    Ranked result + CSV (every value carries its source)
```

The orchestrator is a LangGraph agent. Tools are grouped into: metal tools (LAMMPS), polymer tools (Amber), a curated reference lookup, RAG over software manuals and scientific literature, autonomous web error-search, and a mandatory `complete_experiment` step that ranks results and writes the CSV.

---

## Repository layout

| File | Purpose |
|---|---|
| `atomagents_langgraph_mixed.py` | LangGraph agent: system prompt, tool definitions, control flow, `complete_experiment` |
| `run_yaml_experiment.py` | Launcher — turns a YAML question into a task and runs the agent |
| `lammps_functions.py` | Metal calculations via LAMMPS (lattice, elastic, surface energy, derived density) |
| `amber_functions.py` | Polymer calculations via Amber MD (minimization, heating, NPT density, energy-deformation modulus) |
| `polymer_builder.py` | Builds polymer topology from SMILES (RDKit → antechamber → tleap → packmol) |
| `reference_data.py` | Single source of curated experimental references (metals + polymers), each with its source |
| `example_experiment.yaml` | A ready-to-run comparison (Aluminum vs PLA) |

> Some components referenced by the code are environment-specific and **not included** here (they must be built or provided locally): the RAG knowledge bases (`build_rag.py`, `build_software_rag.py`), the environment checker (`middleware.py`), the software-validation agent (`software_agent.py`), and the interatomic potential files (`potential_repository/`).

---

## Requirements

This is **HPC / research software**. It depends on tools that are *not* pip-installable and must be present in your environment:

- **LAMMPS** (with the Python interface) — for metals
- **Amber / AmberTools** — for polymers
- **Ollama** running locally, serving an open-source LLM (e.g. `qwen2.5`)
- **A GPU** for the MD runs
- Python packages in `requirements.txt` (LangGraph, RDKit, ParmEd, ASE, ChromaDB, …)

On an HPC cluster these typically come from environment modules
(`module load Amber ...`, `module load LAMMPS ...`).

---

## Usage

```bash
# 1. edit the personal/environment paths (see the "Important" note above)
# 2. make sure Ollama is serving your model
#      ollama serve &
#      ollama pull qwen2.5:32b-instruct-q8_0
# 3. place your interatomic potential in  potential_repository/
# 4. run an experiment
python run_yaml_experiment.py example_experiment.yaml
```

The agent will plan the workflow, run the metal and polymer calculations,
evaluate each result, and write a ranked CSV where **every value carries its source**.

---

## Scope & honest limitations

- **This is a research prototype**, built and validated on a specific HPC environment. Running it elsewhere requires editing the paths above and providing the tools.
- **MD-computed moduli are noisy.** The polymer Young modulus is derived from a bulk modulus with an assumed Poisson ratio; deviations from experiment are expected and are *reported*, not hidden. (In one validation run the PLA density landed within ~7% of experiment while the modulus deviated more — and the agent flagged and explained it.)
- **Glass transition (Tg)** is declared as a tool but returns `not_implemented` rather than a disguised experimental value.
- The **RAG knowledge bases** and **potential files** are not shipped; you build/provide them for your own corpus.

The guiding principle throughout: **a computed value is reported with its provenance; a reference value is only ever used for comparison. Nothing is fabricated.**

---

## Acknowledgements

Developed under the **Momentum programme at CSIC**, running on the **Drago HPC cluster** (NVIDIA H200 GPUs).

Inspired by earlier multi-agent work on materials design, but with an independent architecture focused on local open-source models, autonomy on HPC, and critical evaluation of results.

---

## License

_To be defined by the author in accordance with institutional policy._
