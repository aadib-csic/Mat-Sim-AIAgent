#!/lustre/home/admins/aadib/.conda/envs/atomagents/bin/python3
# coding: utf-8
"""
run_yaml_experiment.py

Script universal para lanzar experimentos AtomAgents desde un fichero YAML.

Uso:
    python run_yaml_experiment.py experiment.yaml
"""

import os
import sys
import glob
import logging
import requests
import yaml
from datetime import datetime
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import matplotlib; matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Argumento: fichero YAML
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    print("Uso: python run_yaml_experiment.py <experiment.yaml>")
    sys.exit(1)

yaml_file = sys.argv[1]
if not os.path.exists(yaml_file):
    print(f"Error: fichero no encontrado: {yaml_file}")
    sys.exit(1)

with open(yaml_file) as f:
    config = yaml.safe_load(f)

name     = config.get("name", "experiment")
question = config.get("question", "")
output   = config.get("output", f"code_dir/{name}_results.csv")

if "<FILL" in name or "<FILL" in question:
    print("Error: rellena los campos <FILL> en el YAML antes de ejecutar.")
    sys.exit(1)

metals   = config.get("materials", {}).get("metals", []) or []
polymers = config.get("materials", {}).get("polymers", []) or []

if not metals and not polymers:
    print("Error: añade al menos un material en metals o polymers.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path  = f"logs/{name}_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yaml_experiment")

logger.info("=" * 70)
logger.info(f"AtomAgents — {name}")
logger.info(f"Question: {question}")
logger.info("=" * 70)

# ---------------------------------------------------------------------------
# PASO 1: Cargar modelo en VRAM UNA SOLA VEZ antes de importar el grafo
# ---------------------------------------------------------------------------
OLLAMA_URL      = "http://localhost:11434"
PREFERRED_MODEL = "qwen2.5:32b-instruct-q8_0"
FALLBACK_MODEL  = "qwen2.5:3b"

def ensure_ollama_model() -> str:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        logger.error(f"Ollama no responde: {e}")
        sys.exit(1)

    model = PREFERRED_MODEL if PREFERRED_MODEL in models else FALLBACK_MODEL
    if model not in models:
        logger.error(f"Ningun modelo disponible. Ejecuta: ollama pull {PREFERRED_MODEL}")
        sys.exit(1)

    logger.info(f"Cargando {model} en VRAM...")
    try:
        requests.post(f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "ping", "stream": False},
            timeout=120)
        logger.info(f"Modelo {model} listo en VRAM")
        logger.info("=" * 50)
        logger.info("ENTORNO HPC VERIFICADO:")
        logger.info(f"  Ollama:  OK — {model}")
    except Exception as e:
        logger.warning(f"Warmup fallo: {e}")

    os.environ["OLLAMA_MODEL"] = model
    return model

model_name = ensure_ollama_model()

# ---------------------------------------------------------------------------
# PASO 2: Middleware — verificar LAMMPS y Amber
# ---------------------------------------------------------------------------
need_amber = len(polymers) > 0
from middleware import ensure_environment
env = ensure_environment(need_lammps=True, need_amber=need_amber, need_ollama=False)
env["ollama_model"] = model_name

if not env["lammps"] and len(metals) > 0:
    logger.error("LAMMPS no disponible para simulaciones de metales")
    sys.exit(1)
if not env["amber"] and need_amber:
    logger.error("Amber no disponible para simulaciones de polimeros")
    sys.exit(1)
logger.info(f"  LAMMPS:  {'OK' if env['lammps'] else 'N/A'}")
logger.info(f"  Amber:   {'OK' if env['amber'] else 'N/A'}")
logger.info("=" * 50)

# ---------------------------------------------------------------------------
# PASO 3: Software Agent — validar parametros antes de ejecutar
# ---------------------------------------------------------------------------
from software_agent import validate_experiment
# Suprimir logs de httpx durante la validacion
import logging as _logging
_logging.getLogger("httpx").setLevel(_logging.WARNING)
_logging.getLogger("httpcore").setLevel(_logging.WARNING)
logger.info("=" * 50)
logger.info("SOFTWARE AGENT — validando parametros...")
logger.info("  RAG: Amber 25 manual + LAMMPS docs")
validation = validate_experiment(config)
logger.info(f"Validacion: {validation['status']}")
if validation.get('issues'):
    for issue in validation['issues']:
        logger.warning(f"  ISSUE: {issue}")
if validation.get('corrections'):
    for correction in validation['corrections']:
        logger.info(f"  CORRECCION: {correction}")
if validation['status'] == 'ERROR':
    logger.error("Experimento bloqueado por el software agent")
    sys.exit(1)

# ---------------------------------------------------------------------------
# PASO 4: Verificar ficheros necesarios
# ---------------------------------------------------------------------------
for metal in metals:
    pot = f"potential_repository/{metal['potential']}"
    if not os.path.exists(pot):
        logger.error(f"Potencial no encontrado: {pot}")
        sys.exit(1)
    logger.info(f"  Potencial OK: {pot}")

for polymer in polymers:
    if polymer.get("smiles"):
        logger.info(f"  SMILES OK: {polymer['name']} — se construira topologia desde SMILES")
    else:
        mol2 = f"polymer_mol2/monomer_{polymer['name']}.mol2"
        if not os.path.exists(mol2):
            logger.error(f"Mol2 no encontrado: {mol2}")
            sys.exit(1)
        logger.info(f"  Mol2 OK: {mol2}")

# ---------------------------------------------------------------------------
# PASO 5: Importar sistema — el modelo ya esta en VRAM, no se recarga
# ---------------------------------------------------------------------------
from atomagents_langgraph_mixed import mixed_graph
from langchain_core.messages import HumanMessage
logger.info(f"Sistema listo. Modelo: {model_name}")

# ---------------------------------------------------------------------------
# Construir tarea desde YAML
# ---------------------------------------------------------------------------
def _normalize_props(props):
    """Normaliza el vocabulario de propiedades a un juego canónico, aceptando
    los alias que aparecen en los YAML (young_modulus, modulus -> E, etc.).
    Corrige el bug por el que 'modulus'/'young_modulus' no generaban el paso
    de E porque el generador solo reconocía la cadena exacta 'E'."""
    alias = {
        "e": "E", "young": "E", "young_modulus": "E", "youngs_modulus": "E",
        "modulus": "E", "e_gpa": "E",
        "density": "density", "rho": "density", "density_gcm3": "density",
        "tg": "Tg", "glass_transition": "Tg", "tg_k": "Tg",
        "lattice_constant": "lattice_constant", "lattice": "lattice_constant", "a": "lattice_constant",
        "elastic_constants": "elastic_constants", "elastic": "elastic_constants",
        "kic": "KIc", "fracture_toughness": "KIc",
        "surface_energy_110": "surface_energy_110", "surface_energy": "surface_energy_110",
    }
    out = []
    for p in (props or []):
        out.append(alias.get(str(p).strip().lower(), p))
    return out


def build_task_from_yaml(config: dict) -> str:
    question    = config["question"]
    metals      = config.get("materials", {}).get("metals", []) or []
    polymers    = config.get("materials", {}).get("polymers", []) or []
    analysis    = config.get("analysis", {})
    metric      = analysis.get("metric", config.get("metric", "E_over_rho"))
    objective   = analysis.get("objective", config.get("objective", "maximize"))
    compare_ref = analysis.get("compare_with_rag", True)
    output      = config.get("output", "code_dir/results.csv")

    lines = [f"Scientific question: {question}", "",
             "Compute the following properties, then EVALUATE each against the "
             "literature reference and report honestly (flag deviations >~10% "
             "with a warning; retry once only if it did not converge).", ""]

    # Si la métrica de ranking necesita una propiedad que el YAML no pidió
    # explícitamente, hay que calcularla igualmente (p.ej. E_over_rho necesita
    # densidad Y módulo en AMBOS materiales, aunque el YAML de un material solo
    # liste uno). Se infiere del nombre de la métrica.
    metric_needs = set()
    ml = metric.lower()
    if "rho" in ml or "density" in ml:
        metric_needs.add("density")
    if ml.startswith("e") or "young" in ml or "modulus" in ml or "e_over" in ml:
        metric_needs.add("E")

    def _augment(props):
        p = _normalize_props(props)
        for need in metric_needs:
            if need not in p:
                p = p + [need]
        return p

    step = 1
    for metal in metals:
        mname     = metal["name"]
        pot       = metal["potential"]
        elements  = metal.get("elements", mname)
        structure = metal.get("structure", "fcc")
        guess     = metal.get("lattice_guess", 3.5)
        props     = _augment(metal.get("properties", ["E"]))
        prefix    = f"exp_{mname.lower()}"

        lines.append(f"=== {mname} (metal, LAMMPS) ===")

        need_lattice = any(p in props for p in
                           ["lattice_constant", "elastic_constants", "E", "KIc",
                            "surface_energy_110", "density"])
        if need_lattice:
            lines.append(f"STEP {step}: compute_lattice_constant(material={mname}, potential_file={pot}, pair_coeff_elements={elements}, lat_type={structure}, lat_const_guess={guess}, working_directory={prefix}_lattice, conc_1=100, conc_2=0)")
            step += 1

        # densidad de metal: DERIVADA de la constante de red (no a mano)
        if "density" in props:
            lines.append(f"STEP {step}: compute_metal_density_tool(lattice_constant_A=<value from lattice step>, element={mname}, structure={structure})")
            step += 1

        if any(p in props for p in ["elastic_constants", "E", "KIc"]):
            lines.append(f"STEP {step}: compute_elastic_constants(material={mname}, potential_file={pot}, pair_coeff_elements={elements}, lat_type={structure}, lattice_constant=<value from previous step>, working_directory={prefix}_elastic, conc_1=100, conc_2=0)")
            step += 1

        if "E" in props or "KIc" in props:
            lines.append(f"STEP {step}: compute_young_modulus_tool(material={mname}, C11=<from elastic>, C12=<from elastic>, C44=<from elastic>)")
            step += 1

        if "surface_energy_110" in props or "KIc" in props:
            lines.append(f"STEP {step}: compute_surface_energy(material={mname}, potential_file={pot}, pair_coeff_elements={elements}, lat_type={structure}, lattice_constant=<from lattice step>, plane=110, working_directory={prefix}_surf, conc_1=100, conc_2=0)")
            step += 1

        if "KIc" in props:
            lines.append(f"STEP {step}: compute_fracture_toughness(material={mname}, C11=<from elastic>, C12=<from elastic>, C44=<from elastic>, surface_energy=<from surface step>)")
            step += 1

        lines.append("")

    for polymer in polymers:
        pname        = polymer["name"]
        n_chains     = polymer.get("n_chains", 10)
        chain_length = polymer.get("chain_length", 20)
        temp         = polymer.get("temperature_K", 300.0)
        props        = _augment(polymer.get("properties", ["E"]))
        smiles       = polymer.get("smiles", "")
        prefix       = f"exp_{pname.lower()}"

        lines.append(f"=== {pname} (polymer, Amber) ===")

        # Construir topología UNA vez y reusarla (topology_folder) en los
        # cálculos siguientes -> evita reconstruir antechamber (~4 min).
        topo_folder = f"{prefix}_topology"
        if smiles:
            lines.append(f"STEP {step}: prepare_polymer_topology_tool(polymer_name={pname}, smiles='{smiles}', n_chains={n_chains}, chain_length={chain_length}, working_directory={topo_folder})")
            step += 1

        if "density" in props:
            reuse = f", topology_folder={topo_folder}" if smiles else ""
            lines.append(f"STEP {step}: compute_polymer_density_tool(polymer_type={pname}, working_directory={prefix}_density, n_chains={n_chains}, chain_length={chain_length}, temperature_K={temp}{reuse})")
            step += 1

        if "Tg" in props:
            lines.append(f"STEP {step}: compute_glass_transition_tool(polymer_type={pname}, working_directory={prefix}_Tg, n_chains={n_chains}, chain_length={chain_length})")
            step += 1

        if "E" in props:
            lines.append(f"STEP {step}: compute_polymer_modulus_tool(polymer_type={pname}, working_directory={prefix}_modulus, n_chains={n_chains}, chain_length={chain_length}, temperature_K={temp})")
            step += 1

        lines.append("")

    if compare_ref:
        all_names = [m["name"] for m in metals] + [p["name"] for p in polymers]
        lines.append("=== EVALUATION (compare each COMPUTED value to literature) ===")
        lines.append("For each material, call get_reference_value(material, prop) for the "
                     "properties you computed, compute the error, and decide accept / "
                     "retry-once / flag-with-warning per the rules. The reference value "
                     "is for discussion only — never put it in the results.")
        for mat in all_names:
            lines.append(f"STEP {step}: get_reference_value(material='{mat}', prop='density')  # and other computed props")
            step += 1
        lines.append("")

    lines.append("=== FINAL ANALYSIS ===")
    lines.append(f"STEP {step}: complete_experiment(")
    lines.append(f"    results='<dict: each material -> each property -> "
                 f"{{\"value\": <computed>, \"source\": \"LAMMPS\"|\"LAMMPS_derived\"|\"Amber\", "
                 f"\"warning\": <only if flagged>}}>',")
    lines.append(f"    metric='{metric}',")
    lines.append(f"    objective='{objective}',")
    lines.append(f"    output_file='{output}',")
    lines.append(f"    scientific_summary='<interpretation CONSISTENT with the ranking; "
                 f"state the winner and numbers; mention any flagged values>'")
    lines.append(f")")
    step += 1
    lines.append("")
    lines.append("Execute steps in order. One tool at a time. NEVER invent values. "
                 "NEVER put a reference value into results. complete_experiment is "
                 "MANDATORY and saves the CSV; the run ends automatically after it.")

    return "\n".join(lines)


task_content = build_task_from_yaml(config)
logger.info(f"\nTarea generada (primeros 500 chars):\n{task_content[:500]}...")

# ---------------------------------------------------------------------------
# Ejecutar
# ---------------------------------------------------------------------------
def stream_graph(graph, initial_state, task_name=""):
    try:
        for event in graph.stream(
            initial_state,
            config={"recursion_limit": 100},
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                for msg in node_output.get("messages", []):
                    content    = getattr(msg, "content", "") or ""
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    tool_call_id = getattr(msg, "tool_call_id", None)
                    mname      = getattr(msg, "name", None)
                    if tool_calls:
                        for tc in tool_calls:
                            tc_name = tc.get("name","") if isinstance(tc,dict) else getattr(tc,"name","")
                            tc_args = tc.get("args",{}) if isinstance(tc,dict) else getattr(tc,"args",{})
                            logger.info(f"\n{'='*60}\nLLM -> TOOL: {tc_name}\n  Args: {str(tc_args)[:300]}\n{'='*60}")
                    elif tool_call_id:
                        logger.info(f"\n[RESULTADO] {mname}:\n  {content[:400]}")
                    elif content and "TERMINATE" in content.upper():
                        logger.info(f"\n{'='*60}\n[CONCLUSION]:\n{content}\n{'='*60}")
                    elif content and node_name == "agent":
                        logger.info(f"\n[LLM]: {content[:300]}")
    except Exception as e:
        logger.exception(f"Error en {task_name}")
        raise

logger.info("\nEjecutando experimento...")
stream_graph(
    mixed_graph,
    {"messages": [HumanMessage(content=task_content)], "results": {}, "errors": []},
    task_name=name
)

# Limpiar
for f in glob.glob("run_lammps_*.py"): os.remove(f)

# Verificar resultado. NO se genera un CSV-fantasma desde el YAML: si el
# experimento no produjo CSV, es un FALLO honesto, no un éxito simulado.
if os.path.exists(output):
    logger.info(f"\nEXITO: {output}\n" + open(output).read())
else:
    logger.error(f"CSV NO generado: {output}. El experimento no completó "
                 f"complete_experiment. Esto es un FALLO — no se fabrica un "
                 f"CSV de relleno desde el YAML (sería un resultado vacío "
                 f"disfrazado de éxito). Revisa el log del agente arriba.")

# ---------------------------------------------------------------------------
# LIMPIEZA — solo scripts y logs temporales de LAMMPS.
# NO se borran los directorios exp_* : contienen las topologías y trayectorias
# (reutilizables y necesarias para inspección). El bug anterior hacía
# shutil.rmtree(exp_*) y arrasaba la topología recién construida.
# ---------------------------------------------------------------------------
for f in glob.glob("run_lammps_*.py"):
    try: os.remove(f)
    except OSError: pass
for f in glob.glob("log.lammps.*"):
    try: os.remove(f)
    except OSError: pass
logger.info("Limpieza: scripts/logs temporales de LAMMPS eliminados "
            "(directorios exp_* conservados).")

logger.info(f"\nLog: {log_path}")
