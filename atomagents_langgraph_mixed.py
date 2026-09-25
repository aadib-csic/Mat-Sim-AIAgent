#!/usr/bin/env python
# coding: utf-8
"""
atomagents_langgraph_mixed.py - Sistema AtomAgents con LangGraph
"""

import os
import sys
import json
import logging
import subprocess

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

from typing import TypedDict, Annotated, Literal, Any
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Importar funciones
from lammps_functions import *
from amber_functions import *
from polymer_builder import create_polymer_builder_tool
prepare_polymer_topology_tool = create_polymer_builder_tool()

logger = logging.getLogger("atomagents_mixed")

# ---------------------------------------------------------------------------
# ESTADO
# ---------------------------------------------------------------------------
class MixedState(TypedDict):
    messages: Annotated[list, add_messages]
    results: dict
    errors: list

# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------


def _fix_pair_coeff_elements(pair_coeff_elements: str, conc_2: int,
                             material: str = "X") -> str:
    """BLINDAJE: ajusta el nº de elementos del pair_coeff al nº de tipos de
    átomo. Sistema puro (conc_2=0) => 1 tipo => 1 elemento. El LLM a veces
    duplica ('Al Al'), lo que rompe LAMMPS con 'Number of element to type
    mappings does not match number of atom types'. Corrige de forma
    determinista preservando el orden. Devuelve el string corregido."""
    raw = [e for e in pair_coeff_elements.strip().split() if e and e != 'NULL']
    n_types = 1 if conc_2 == 0 else 2
    if len(raw) >= n_types:
        elements = raw[:n_types]
    elif raw:
        elements = [raw[0]] * n_types
    else:
        elements = [material] * n_types
    fixed = " ".join(elements)
    if len(raw) != n_types:
        logger.warning(f"pair_coeff_elements ajustado: '{pair_coeff_elements}' "
                       f"-> '{fixed}' (conc_2={conc_2} => {n_types} atom type(s))")
    return fixed


@tool
def compute_lattice_constant(
    material: str,
    potential_file: str,
    pair_coeff_elements: str,
    lat_type: str,
    lat_const_guess: float,
    working_directory: str,
    conc_1: int = 100,
    conc_2: int = 0,
    num_cpus: int = 4,
) -> str:
    """Compute equilibrium lattice constant using LAMMPS EAM potential."""
    try:
        create_working_folder(working_directory)
        pair_coeff_elements = _fix_pair_coeff_elements(
            pair_coeff_elements, conc_2, material)
        create_potential_file(working_directory, "eam/alloy",
            f"* * ../potential_repository/{potential_file} {pair_coeff_elements}")
        elements = pair_coeff_elements.split()
        el1 = elements[0] if len(elements) > 0 else 'X'
        el2 = elements[1] if len(elements) > 1 else 'Y'
        result = lattice_constant_simulation(
            working_folder=working_directory, lat_type=lat_type,
            lat_const_guess=lat_const_guess, conc_1=conc_1, conc_2=conc_2,
            mtp_pot=False, num_cpus=num_cpus, element_1=el1, element_2=el2)
        return f"Lattice constant for {material}: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def compute_elastic_constants(
    material: str,
    potential_file: str,
    pair_coeff_elements: str,
    lat_type: str,
    lattice_constant: float,
    working_directory: str,
    conc_1: int = 100,
    conc_2: int = 0,
) -> str:
    """Compute elastic constants C11, C12, C44 using LAMMPS."""
    try:
        create_working_folder(working_directory)
        pair_coeff_elements = _fix_pair_coeff_elements(
            pair_coeff_elements, conc_2, material)
        create_potential_file(working_directory, "eam/alloy",
            f"* * ../potential_repository/{potential_file} {pair_coeff_elements}")
        elements = pair_coeff_elements.split()
        el1 = elements[0] if len(elements) > 0 else 'X'
        el2 = elements[1] if len(elements) > 1 else 'Y'
        result = elastic_constant_simulation(
            working_folder=working_directory, lat_type=lat_type,
            a=lattice_constant, conc_1=conc_1, conc_2=conc_2,
            mtp_pot=False, element_1=el1, element_2=el2)
        return f"Elastic constants for {material}: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def compute_surface_energy(
    material: str,
    potential_file: str,
    pair_coeff_elements: str,
    lat_type: str,
    lattice_constant: float,
    plane: str,
    working_directory: str,
    conc_1: int = 100,
    conc_2: int = 0,
) -> str:
    """Compute surface energy for a crystallographic plane using LAMMPS."""
    try:
        create_working_folder(working_directory)
        pair_coeff_elements = _fix_pair_coeff_elements(
            pair_coeff_elements, conc_2, material)
        create_potential_file(working_directory, "eam/alloy",
            f"* * ../potential_repository/{potential_file} {pair_coeff_elements}")
        elements = pair_coeff_elements.split()
        el1 = elements[0] if len(elements) > 0 else 'X'
        el2 = elements[1] if len(elements) > 1 else 'Y'
        if plane == '110':
            ox, oy, oz = [1,1,0], [0,0,1], [1,-1,0]
        elif plane == '111':
            ox, oy, oz = [1,1,1], [1,-1,0], [1,1,-2]
        else:
            return f"Error: plane must be '110' or '111'"
        result = surface_energy_simulation(
            working_folder=working_directory, lat_type=lat_type,
            lat_const=lattice_constant, conc_1=conc_1, conc_2=conc_2,
            mtp_pot=False, orient_x=ox, orient_y=oy, orient_z=oz,
            surface_orientation=ox, surface_dir='x',
            element_1=el1, element_2=el2)
        return f"Surface energy [{plane}] for {material}: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def compute_young_modulus_tool(material: str, C11: float, C12: float, C44: float) -> str:
    """Compute Young modulus E using Voigt-Reuss-Hill."""
    try:
        result = compute_young_modulus(C11, C12, C44)
        return f"Elastic moduli for {material}: {json.dumps(result)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def compute_fracture_toughness(
    material: str, C11: float, C12: float, C44: float,
    surface_energy: float, crack_system: str = "110_001_1-10",
) -> str:
    """Compute fracture toughness KIc using LEFM."""
    try:
        crack_systems = {
            "110_001_1-10": ([1,1,0], [0,0,1], [1,-1,0]),
            "111_110_1-12": ([1,1,1], [1,1,0], [1,-1,-2]),
            "100_010_001":  ([1,0,0], [0,1,0], [0,0,1]),
        }
        ox, oy, oz = crack_systems.get(crack_system, crack_systems["110_001_1-10"])
        result = critical_stress_intensity_cleavage(
            orient_x=ox, orient_y=oy, orient_z=oz,
            c11=C11, c12=C12, c44=C44, gamma_s=surface_energy)
        return f"Fracture toughness KIc for {material}: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def compute_polymer_density_tool(
    polymer_type: str, working_directory: str,
    n_chains: int = 10, chain_length: int = 20,
    temperature_K: float = 300.0, n_steps: int = 500000,
    topology_folder: str = "",
) -> str:
    """Compute density using Amber/GAFF2. If topology_folder is provided,
    reuses pre-built topology from prepare_polymer_topology_tool."""
    result = compute_polymer_density(
        polymer_type=polymer_type, working_folder=working_directory,
        n_chains=n_chains, chain_length=chain_length,
        temperature_K=temperature_K, n_steps_equil=n_steps,
        topology_folder=topology_folder if topology_folder else None)
    return result


@tool
def compute_glass_transition_tool(
    polymer_type: str, working_directory: str,
    n_chains: int = 10, chain_length: int = 20,
    T_high_K: float = 600.0, T_low_K: float = 100.0,
    n_T_steps: int = 10, n_steps_per_T: int = 200000,
) -> str:
    """Compute glass transition temperature Tg by MD cooling ramp.
    NOTE: not implemented yet — returns status='not_implemented' rather than
    a disguised experimental value. Use get_reference_value for a literature
    Tg (discussion only)."""
    try:
        result = compute_glass_transition(
            polymer_type=polymer_type, working_folder=working_directory,
            n_chains=n_chains, chain_length=chain_length,
            T_high_K=T_high_K, T_low_K=T_low_K,
            n_T_steps=n_T_steps, n_steps_per_T=n_steps_per_T)
        return result
    except Exception as e:
        return json.dumps({"error": str(e), "polymer": polymer_type})


@tool
def compute_polymer_modulus_tool(
    polymer_type: str, working_directory: str,
    n_chains: int = 10, chain_length: int = 20,
    temperature_K: float = 300.0, poisson_ratio: float = 0.40,
) -> str:
    """Compute Young modulus E by REAL MD (hydrostatic energy-deformation).
    Measures bulk modulus K from the energy curvature under box strain, then
    derives E = 3K(1-2*nu) with the assumed poisson_ratio reported explicitly.
    Returns K_GPa, E_GPa, fit_R2 (quality flag), converged, source='Amber'.
    A high error vs reference with high R2 is a method/nu limitation, not a
    convergence failure — evaluate accordingly."""
    try:
        result = compute_polymer_modulus(
            polymer_type=polymer_type, working_folder=working_directory,
            n_chains=n_chains, chain_length=chain_length,
            temperature_K=temperature_K, poisson_ratio=poisson_ratio)
        return result
    except Exception as e:
        return json.dumps({"error": str(e), "polymer": polymer_type})


@tool
def get_polymer_reference_tool(polymer_type: str) -> str:
    """Get experimental reference values."""
    return get_polymer_reference(polymer_type)


@tool
def get_reference_value(material: str, prop: str = "") -> str:
    """Get a CURATED literature reference value for a material property, WITH
    its source. Works for metals (Al, Ni, Cu, W) and polymers (PE, PP, PS,
    PMMA, PLA). Use this ONLY during EVALUATION, to compare a COMPUTED value
    against literature. NEVER put a reference value into the results/CSV —
    reference is for discussion only; the ranked value must be the computed one.

    Args:
        material: symbol/code, e.g. 'Al' or 'PLA'.
        prop: property, e.g. 'density', 'E', 'Tg', 'lattice_constant'. Empty
              returns all properties for the material.
    Returns JSON with value, unit, and source (or status='not_found')."""
    from reference_data import get_reference
    return json.dumps(get_reference(material, prop if prop else None))


@tool
def compute_metal_density_tool(lattice_constant_A: float, element: str,
                               structure: str = "") -> str:
    """Compute a metal's DENSITY, DERIVED from its computed lattice constant:
    rho = n*M/(N_A*a^3). This is a calculated result (source='LAMMPS_derived'),
    NOT a table lookup. Call this AFTER compute_lattice_constant, passing the
    lattice constant it returned. Molar mass and (if omitted) structure are
    taken from curated data. Use for metals when density is needed (e.g. for
    specific-stiffness E/rho)."""
    try:
        result = compute_metal_density(
            lattice_constant_A=lattice_constant_A, element=element,
            structure=structure if structure else None)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e), "element": element})


# ---------------------------------------------------------------------------
# CONFIGURACIÓN LLM
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"

# ===========================================================================
# TOOL DE VALIDACIÓN DE INPUTS (añadida sin eliminar nada)
# ===========================================================================

@tool
def validate_polymer_inputs_tool(
    polymer_type: str,
    working_directory: str,
    n_chains: int = 10,
    chain_length: int = 20,
    temperature_K: float = 300.0,
) -> str:
    """[VALIDATION] Check polymer simulation inputs before running.

    Args:
        polymer_type: 'PE', 'PP', 'PS', or 'PMMA'
        working_directory: Folder for the simulation
        n_chains: Number of chains
        chain_length: Monomers per chain
        temperature_K: Temperature in Kelvin
    """
    import os
    errors = []
    warnings = []
    
    # Verificar polymer_type
    valid_polymers = ["PE", "PP", "PS", "PMMA"]
    if polymer_type.upper() not in valid_polymers:
        errors.append(f"Invalid polymer: {polymer_type}. Valid: {valid_polymers}")
    
    # Verificar mol2
    mol2_path = f"polymer_mol2/monomer_{polymer_type.upper()}.mol2"
    if not os.path.exists(mol2_path):
        errors.append(f"mol2 file not found: {mol2_path}")
    else:
        # Verificar que tiene el formato correcto
        with open(mol2_path, 'r') as f:
            content = f.read()
        if "@<TRIPOS>MOLECULE" not in content:
            warnings.append("mol2 file may be missing MOLECULE section")
        if "GAFF2" not in content:
            warnings.append("mol2 file may not be GAFF2 format")
    
    # Verificar parámetros
    if n_chains < 1:
        errors.append(f"n_chains must be >= 1 (got: {n_chains})")
    if chain_length < 2:
        errors.append(f"chain_length must be >= 2 (got: {chain_length})")
    if temperature_K < 50 or temperature_K > 1000:
        errors.append(f"temperature_K must be between 50 and 1000 K")
    
    if errors:
        return json.dumps({"valid": False, "errors": errors, "warnings": warnings})
    return json.dumps({"valid": True, "errors": [], "warnings": warnings})

# ===========================================================================
# RAG DE SOFTWARE
# ===========================================================================

@tool
def query_scientific_rag_tool(query: str) -> str:
    """
    Query the scientific RAG database (CRC Handbook, papers) for experimental
    material properties: density, Young modulus, fracture toughness, melting point.
    Use this for: experimental reference values of metals and polymers,
    NOT for Amber/LAMMPS commands (use query_software_rag_tool for that).
    Examples:
    - "aluminum Young modulus density experimental"
    - "fracture toughness steel copper"
    - "polymer glass transition temperature PLA PMMA"
    """
    from rag_tool import query_rag
    return query_rag(query)


@tool
def query_software_rag_tool(query: str) -> str:
    """[RAG] Query software documentation to learn how to use LAMMPS, Amber, etc.

    Use this tool when you need to know:
    - How to set up a simulation
    - What commands to use
    - Best practices for specific software

    Args:
        query: Question about software usage e.g. 'How to compute elastic constants?'
    """
    try:
        import os
        import json

        rag_path = "software_manuals/chroma_db"
        if not os.path.exists(rag_path):
            return json.dumps({
                "error": "Software RAG not found",
                "suggestion": "Run: python build_software_rag.py",
                "fallback": "I will use my built-in knowledge."
            })

        # software_agent ya tiene BGE-M3 + BM25 (retrieval híbrido)
        from software_agent import query_software_manual
        return query_software_manual(query)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "suggestion": "Check that software RAG is built",
            "fallback": "I will use my built-in knowledge."
        })


@tool
def search_error_online(error_message: str, software: str = "lammps") -> str:
    """Search the WEB for a simulation error when the local RAG did not help.
    Queries official docs (LAMMPS/Amber) and DuckDuckGo. Use this to diagnose
    a real error message (e.g. a LAMMPS/pmemd error) AFTER the local RAG failed
    and BEFORE giving up. Read the snippets, then FIX the tool arguments and
    retry ONCE. If it still fails, mark the property failed and move on — do NOT
    call this repeatedly for the same error.

    Args:
        error_message: the exact error text to look up.
        software: 'lammps' or 'amber' (chooses which official docs to prefer).
    """
    import json as _json
    try:
        import requests
        from urllib.parse import quote_plus
    except Exception as e:
        return _json.dumps({"status": "unavailable", "error": f"requests missing: {e}"})

    headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
    results = []

    # 1) DuckDuckGo HTML — restringido al dominio de doc del software
    site = "docs.lammps.org" if software.lower().startswith("l") else "ambermd.org"
    q = f"{error_message} site:{site}"
    try:
        r = requests.get("https://html.duckduckgo.com/html/",
                         params={"q": q}, headers=headers, timeout=12)
        import re
        # extraer snippets de resultado (texto de los <a class="result__a"> y snippets)
        snippets = re.findall(r'result__snippet[^>]*>(.*?)</a>', r.text, re.S)
        links = re.findall(r'result__a[^>]*href="([^"]+)"', r.text)
        for sn, ln in list(zip(snippets, links))[:4]:
            clean = re.sub(r'<[^>]+>', '', sn).strip()
            if clean:
                results.append({"source": site, "url": ln[:200], "text": clean[:400]})
    except Exception as e:
        results.append({"source": "duckduckgo", "error": str(e)})

    # 2) Búsqueda general (sin site:) por si el error es común en foros
    if len(results) < 2:
        try:
            r = requests.get("https://html.duckduckgo.com/html/",
                             params={"q": f"{software} {error_message}"},
                             headers=headers, timeout=12)
            import re
            snippets = re.findall(r'result__snippet[^>]*>(.*?)</a>', r.text, re.S)
            for sn in snippets[:3]:
                clean = re.sub(r'<[^>]+>', '', sn).strip()
                if clean:
                    results.append({"source": "web", "text": clean[:400]})
        except Exception as e:
            results.append({"source": "web", "error": str(e)})

    if not results:
        return _json.dumps({"status": "no_results", "query": error_message,
                            "advice": "No online hits. Mark the property failed "
                                      "and continue; do not retry the same call."})
    return _json.dumps({"status": "ok", "query": error_message,
                        "software": software, "results": results,
                        "advice": "Use these to FIX the tool arguments and retry "
                                  "ONCE. If it still fails, mark failed and move on."},
                       indent=2)




# ---------------------------------------------------------------------------
# TOOL FINAL: complete_experiment — análisis + ranking + CSV en un solo paso
# ---------------------------------------------------------------------------
@tool
def complete_experiment(
    results: Any,
    metric: str = "E_over_rho",
    objective: str = "maximize",
    output_file: str = "code_dir/results.csv",
    scientific_summary: str = "",
) -> str:
    """
    MANDATORY FINAL TOOL — call this after all simulations and evaluation.
    Ranks materials by the metric and saves a CSV. The scientific_summary
    MUST be consistent with the ranking (do not claim a material is best if
    the ranking says otherwise).

    PROVENANCE (important): every property value should be an object carrying
    its origin, e.g.:
        {"Al":  {"density_gcm3": {"value": 2.70, "source": "LAMMPS_derived"},
                 "E_GPa":       {"value": 78.2, "source": "LAMMPS"}},
         "PLA": {"density_gcm3": {"value": 1.14, "source": "Amber"},
                 "E_GPa": {"value": 1.57, "source": "Amber",
                           "warning": "55% below reference; R2=0.998 so the fit is "
                                      "sound — likely finite-size + assumed nu. "
                                      "Recommend more chains / measuring nu."}}}
    A plain number is still accepted (source recorded as 'unspecified'), but
    ALWAYS prefer the object form. NEVER pass a reference/literature value as a
    computed result — reference is for discussion only.

    Args:
        results: dict/JSON. Per material, per property: either an object
                 {"value","source",["warning"]} or a bare number.
        metric: property to rank by ("E_over_rho","E_GPa","density_gcm3","KIc").
        objective: "maximize" or "minimize".
        output_file: path to save CSV.
        scientific_summary: interpretation, CONSISTENT with the ranking.
    """
    import csv, json, os, math

    try:
        if isinstance(results, str):
            data = json.loads(results)
        elif isinstance(results, dict):
            data = results
        else:
            data = json.loads(str(results))
    except Exception as e:
        return json.dumps({"error": f"Could not parse results: {e}"})

    if not data:
        return json.dumps({"error": "No results provided. Pass results as JSON."})

    # --- Normalizar a: values[mat][prop] = float ; meta[mat][prop] = {source,warning}
    # NO hay aliases de blanqueo (density_experimental->density, E_calculated->E):
    # eso permitía colar valores de tabla como calculados. Se elimina.
    values = {}   # mat -> {prop: float}
    meta = {}     # mat -> {prop: {"source":..., "warning":...}}
    warnings_all = []  # (mat, prop, warning, source)

    def _unwrap(v):
        """Devuelve (float|None, source, warning) de un valor que puede ser
        objeto {value,source,warning} o número plano."""
        if isinstance(v, dict):
            val = v.get("value")
            src = v.get("source", "unspecified")
            warn = v.get("warning")
            try:
                val = float(val) if val is not None else None
            except (TypeError, ValueError):
                val = None
            return val, src, warn
        try:
            return float(v), "unspecified", None
        except (TypeError, ValueError):
            return None, "unspecified", None

    for mat, props in data.items():
        if not isinstance(props, dict):
            continue
        values[mat] = {}
        meta[mat] = {}
        for prop, raw in props.items():
            val, src, warn = _unwrap(raw)
            if val is not None:
                values[mat][prop] = val
                meta[mat][prop] = {"source": src}
                if warn:
                    meta[mat][prop]["warning"] = warn
                    warnings_all.append((mat, prop, warn, src))

    # --- Métricas derivadas (E/rho, KIc/E) a partir de los valores calculados
    for mat, props in values.items():
        E = props.get("E_GPa")
        rho = props.get("density_gcm3")
        KIc = props.get("KIc_MPa_sqrt_m")
        if E and rho and rho > 0:
            props["E_over_rho"] = round(E / rho, 3)
            # la métrica derivada hereda advertencia si algún operando la tiene
            inh = [w for (m, p, w, s) in warnings_all if m == mat and p in ("E_GPa", "density_gcm3")]
            meta[mat]["E_over_rho"] = {"source": "derived"}
            if inh:
                meta[mat]["E_over_rho"]["warning"] = "depends on a flagged value: " + inh[0]
        if KIc and E and E > 0:
            props["KIc_over_E"] = round(KIc / E, 5)
            meta[mat]["KIc_over_E"] = {"source": "derived"}

    # --- Ranking
    ranked = []
    for mat, props in values.items():
        val = props.get(metric)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            flagged = "warning" in meta[mat].get(metric, {})
            ranked.append((mat, float(val), flagged))
    ranked.sort(key=lambda x: x[1], reverse=(objective == "maximize"))
    optimal = ranked[0][0] if ranked else "unknown"

    # --- CSV con columnas de valor + fuente + advertencia
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
                exist_ok=True)
    all_props = sorted({p for mat in values for p in values[mat]})
    fieldnames = ["material"] + all_props + \
                 [f"{p}__source" for p in all_props] + ["warnings"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for mat in [m for m, _, _ in ranked] + [m for m in values if m not in [r[0] for r in ranked]]:
            row = {"material": mat}
            for p in all_props:
                if p in values[mat]:
                    v = values[mat][p]
                    row[p] = round(v, 4) if isinstance(v, float) else v
                    row[f"{p}__source"] = meta[mat].get(p, {}).get("source", "")
            mat_warns = [f"{p}: {meta[mat][p]['warning']}"
                         for p in meta[mat] if "warning" in meta[mat][p]]
            row["warnings"] = " | ".join(mat_warns)
            writer.writerow(row)

    # --- Ranking con marca de flagged
    ranking_out = [{"material": m, "value": round(v, 4), "flagged": bool(fl)}
                   for m, v, fl in ranked]

    logger.info(f"Experiment complete. Optimal: {optimal}. CSV: {output_file}")
    if warnings_all:
        logger.warning(f"{len(warnings_all)} flagged value(s): "
                       + "; ".join(f"{m}.{p}" for m, p, _, _ in warnings_all))

    return json.dumps({
        "status": "completed",
        "optimal_material": optimal,
        "metric": metric,
        "objective": objective,
        "ranking": ranking_out,
        "flagged_values": [{"material": m, "property": p, "warning": w, "source": s}
                           for (m, p, w, s) in warnings_all],
        "csv_saved": output_file,
        "scientific_summary": scientific_summary,
        "n_materials": len(values),
    }, indent=2)

ALL_TOOLS = [
    complete_experiment,
    prepare_polymer_topology_tool,
    query_software_rag_tool,
    query_scientific_rag_tool,
    search_error_online,              # NEW: web search for errors (docs + DDG)
    # --- metals (LAMMPS) ---
    compute_lattice_constant,
    compute_elastic_constants,
    compute_surface_energy,
    compute_young_modulus_tool,
    compute_fracture_toughness,
    compute_metal_density_tool,       # NEW: density derived from lattice constant
    # --- polymers (Amber) ---
    compute_polymer_density_tool,
    compute_glass_transition_tool,
    compute_polymer_modulus_tool,
    # --- reference / evaluation ---
    get_polymer_reference_tool,
    get_reference_value,              # NEW: unified curated reference (metals+polymers)
]


MAIN_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b-instruct-q8_0")

llm = ChatOllama(
    model=MAIN_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
    num_ctx=32768,
    num_predict=4096,
)



# ---------------------------------------------------------------------------
# SISTEMA PROMPT
# ---------------------------------------------------------------------------

llm_with_tools = llm.bind_tools(ALL_TOOLS)
tool_node = ToolNode(ALL_TOOLS)

SYSTEM_PROMPT = """You are AtomAgents, a physics-aware AI agent that acts like a computational materials researcher. You COMPUTE properties with the right software, EVALUATE the results against the literature, and REPORT honestly — including when a result is imperfect.

=== GOLDEN RULE ===
The value that goes into the results/ranking is ALWAYS the one you COMPUTED
(LAMMPS for metals, Amber for polymers). A reference/literature value is ONLY
for comparison and discussion — it NEVER replaces a computed value, and never
goes into the ranked results as if it were computed.

=== TOOLS (these are the ONLY tools that exist) ===
METALS (LAMMPS):
- compute_lattice_constant: equilibrium lattice parameter (EAM)
- compute_elastic_constants: C11, C12, C44
- compute_young_modulus_tool: E, G, B, nu from C11/C12/C44
- compute_metal_density_tool: density DERIVED from the computed lattice constant
  (call AFTER compute_lattice_constant, pass its lattice constant)
- compute_surface_energy, compute_fracture_toughness

POLYMERS (Amber/GAFF2):
- prepare_polymer_topology_tool: build topology from SMILES
- compute_polymer_density_tool: density via NPT MD (reuse topology via topology_folder)
- compute_polymer_modulus_tool: Young modulus via energy-deformation MD
  (returns K_GPa, E_GPa, fit_R2, converged; E uses an assumed poisson_ratio)
- compute_glass_transition_tool: Tg (NOT implemented yet -> returns not_implemented)

REFERENCE / EVALUATION:
- get_reference_value: curated literature value (metals + polymers) WITH source
- get_polymer_reference_tool: polymer experimental data
- query_software_rag_tool: how to set up / fix a simulation (Amber/LAMMPS manuals)
- query_scientific_rag_tool: methodological context from papers (NOT for hard numbers)
- search_error_online: search the WEB (official docs + DuckDuckGo) for a real
  error message when the local RAG did not help. Use to diagnose, fix the
  arguments, and retry ONCE — never call it repeatedly for the same error.

=== WHEN A TOOL FAILS (do NOT loop) ===
If a simulation tool returns status=failed or an error:
1. Read the error. If the fix is obvious from the message, correct the tool
   ARGUMENTS and retry once.
2. If not obvious, call query_software_rag_tool once; if that does not help,
   call search_error_online once with the exact error message.
3. Apply what you learned and retry ONCE more.
4. If it STILL fails, STOP calling that tool. Record that property as failed
   (value null, with the error text as its warning) and MOVE ON to the next
   property or to complete_experiment. NEVER call the same tool with the same
   arguments more than ~3 times. NEVER fabricate a value to get past a failure.

FINAL:
- complete_experiment: MANDATORY last step — ranks by the metric and saves CSV.

=== WORKFLOW ===
1. For each material, choose the correct software and COMPUTE the requested
   properties. Metals -> LAMMPS tools. Polymers -> Amber tools.
   - For metal density: first compute_lattice_constant, then compute_metal_density_tool.
   - For polymer E/density on the same system: build topology once, reuse it.
   - If a simulation setup fails, call query_software_rag_tool to diagnose, then retry.
2. EVALUATE each computed value:
   - Call get_reference_value to get the literature value (with its source).
   - Compute the relative error vs the reference.
   - If error <= ~10%: accept the value.
   - If error > ~10%: look at the quality flags (converged, fit_R2).
       * If the calculation did NOT converge (density converged=false, or
         modulus fit_R2<0.95): RETRY ONCE with more sampling (more chains
         and/or more steps).
       * If after that one retry it still deviates, OR if it already converged
         well (e.g. modulus with high fit_R2 but large error): DO NOT keep
         retrying. This is a method/parameter limitation (finite size, force
         field, assumed poisson ratio), not a convergence bug.
   - Either way, ALWAYS continue to the end. A deviating value is not fatal:
     keep it (it is your computed result) but attach a 'warning' describing the
     likely cause and a recommendation for future work.
3. Call complete_experiment with ALL computed values, each carrying provenance:
       {"value": <computed number>, "source": "LAMMPS"|"LAMMPS_derived"|"Amber",
        "warning": "<only if flagged>"}
   Rank by the requested metric.

=== EVALUATION / REPORTING RULES ===
- NEVER invent a numerical value. Every number must come from a tool.
- NEVER put a reference value where a computed value belongs.
- A computed value that deviates from the literature is REPORTED WITH A WARNING,
  not hidden and not replaced. Detecting and explaining a limitation is a valid,
  valuable scientific outcome.
- The scientific_summary you write MUST be consistent with the ranking numbers.
  If the ranking says material X is best on the metric, do NOT claim another is
  better. State the winner, the numbers, and any caveats (flagged values).
- One tool call at a time.

=== EXAMPLE (E/rho of a metal vs a polymer) ===
compute_lattice_constant(Al) -> a=4.05
compute_metal_density_tool(4.05, "Al") -> rho=2.70 (LAMMPS_derived)
compute_elastic_constants(Al) -> C11,C12,C44 ; compute_young_modulus_tool -> E=78 (LAMMPS)
get_reference_value("Al","E") -> 70 GPa (CRC) ; error ~11%, converged -> accept, minor note
prepare_polymer_topology_tool(PLA) ; compute_polymer_density_tool(PLA, reuse topology) -> 1.14 (Amber)
get_reference_value("PLA","density") -> 1.24 ; error 8% -> accept
compute_polymer_modulus_tool(PLA) -> E=1.57, fit_R2=0.998 (Amber)
get_reference_value("PLA","E") -> 3.5 ; error 55% BUT fit_R2=0.998 -> converged, so do
   NOT retry blindly; attach warning: "R2=0.998 so fit is sound; deviation likely from
   finite size + assumed nu=0.40; recommend more chains or measuring nu."
complete_experiment(results={... each value with value/source/warning ...},
   metric="E_over_rho", objective="maximize",
   scientific_summary="Al ranks first on E/rho (28.9 vs 1.4). PLA's modulus is
   flagged (see warning); its density (1.14) matches literature within 8%.")
"""

# ---------------------------------------------------------------------------
# GRAFO
# ---------------------------------------------------------------------------

def _count_repeated_failures(messages):
    """Cuenta cuántas veces SEGUIDAS la misma tool ha devuelto error/failed al
    final del historial. Devuelve (tool_name, n_consecutive_failures)."""
    from langchain_core.messages import ToolMessage
    import json as _json
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_msgs:
        return None, 0
    def _is_fail(m):
        c = getattr(m, "content", "") or ""
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        cl = c.lower()
        return ('"status": "failed"' in cl or '"error"' in cl
                or cl.strip().startswith("error:"))
    last_name = getattr(tool_msgs[-1], "name", "")
    n = 0
    for m in reversed(tool_msgs):
        if getattr(m, "name", "") == last_name and _is_fail(m):
            n += 1
        else:
            break
    return last_name, n


def agent_node(state: MixedState) -> MixedState:
    from langchain_core.messages import SystemMessage
    messages = state["messages"]
    # El system prompt debe estar SIEMPRE presente, no solo en el primer turno.
    # (Bug anterior: 'if len==1' lo ponía solo al inicio y se caía del contexto
    #  a partir del turno 2, dejando al LLM sin sus reglas de evaluación.)
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # CORTA-BUCLES: si la misma tool ha fallado 3+ veces seguidas, inyectar
    # una instrucción dura para que el LLM PARE de reintentarla y siga.
    fail_tool, n_fail = _count_repeated_failures(messages)
    if n_fail >= 3:
        stop_msg = SystemMessage(content=(
            f"STOP: the tool '{fail_tool}' has failed {n_fail} times in a row "
            f"with the same kind of error. Do NOT call '{fail_tool}' again now. "
            f"Record the affected property as failed (value null, error text as "
            f"its warning) and either move to the next property or call "
            f"complete_experiment with what you already have. Do NOT fabricate "
            f"any value."))
        messages = messages + [stop_msg]
        logger.warning(f"CORTA-BUCLES activado: '{fail_tool}' falló {n_fail}x — "
                       f"forzando al agente a continuar")

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MixedState) -> Literal["tools", "agent", "end"]:
    last = state["messages"][-1]
    messages = state["messages"]

    # ¿Se ha llamado ya complete_experiment? (recorre el historial)
    tools_called = set()
    complete_experiment_done = False
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            tools_called.add(name)
    # complete_experiment se considera hecho cuando su ToolMessage ya volvió
    # (es decir, hay un resultado de esa tool en el historial), no solo cuando
    # se pidió. Buscamos el ToolMessage correspondiente.
    from langchain_core.messages import ToolMessage
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "complete_experiment":
            complete_experiment_done = True
            break

    # 1) Si el último mensaje pide tools -> ejecutarlas.
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # 2) Si complete_experiment ya se ejecutó, TERMINAR limpio.
    #    (No depende de que el LLM escriba la palabra 'TERMINATE'; el bug
    #     anterior sólo terminaba por coincidencia de subcadena.)
    if complete_experiment_done:
        return "end"

    # 3) El LLM emitió texto sin tool_calls y aún no ha completado el
    #    experimento. Volver al AGENTE (no a tools, que fallaría sin
    #    tool_calls) para que reconsidere y llame a la siguiente tool.
    #    El recursion_limit del .invoke() es la red de seguridad dura.
    return "agent"


def build_mixed_graph():
    graph = StateGraph(MixedState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue,
        {"tools": "tools", "agent": "agent", "end": END}
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


mixed_graph = build_mixed_graph()
