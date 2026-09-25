#!/usr/bin/env python
# coding: utf-8
"""
amber_functions.py

Funciones Amber puras para simulación de polímeros.
"""

import os
import re
import json
import subprocess
import numpy as np
import logging

logger = logging.getLogger("amber_functions")

# ---------------------------------------------------------------------------
# PATHS DE AMBER
# ---------------------------------------------------------------------------
AMBER_BIN = "/lustre/opt/foss/0.4/software/Amber/24.0-foss-2023b-AmberTools-25.0-CUDA-12.4.0/bin"
AMBERHOME = "/lustre/opt/foss/0.4/software/Amber/24.0-foss-2023b-AmberTools-25.0-CUDA-12.4.0"

# ---------------------------------------------------------------------------
# BASE DE DATOS DE MONÓMEROS
# ---------------------------------------------------------------------------
# POLYMER_DB: fuente ÚNICA de referencia en reference_data.py.
# Aquí sólo se AÑADEN campos específicos de CONSTRUCCIÓN de topología
# (monomer_name para la ruta mol2 antigua). Los valores experimentales
# (density_exp_gcm3, E_exp_GPa, Tg_exp_K) vienen ya de reference_data.
from reference_data import POLYMER_DB as _REF_POLYMER_DB

_MONOMER_BUILD_INFO = {
    "PE":   {"monomer_name": "ETH"},
    "PP":   {"monomer_name": "PRP"},
    "PS":   {"monomer_name": "STY"},
    "PMMA": {"monomer_name": "MMA"},
    "PLA":  {"monomer_name": "PLA"},
}

POLYMER_DB = {
    code: {**_REF_POLYMER_DB[code], **_MONOMER_BUILD_INFO.get(code, {})}
    for code in _REF_POLYMER_DB
}

POLYMER_MOL2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polymer_mol2")

# Nota: polímeros con SMILES generan mol2 dinámicamente via polymer_builder.py
# Solo los polímeros con mol2 preexistente están aquí
# Mol2 preexistentes — vacío: todos los polímeros usan RDKit pipeline
# Ver polymer_builder.py POLYMER_REPEAT_SMILES para los SMILES de oligómeros
POLYMER_MOL2_FILES = {}

# ---------------------------------------------------------------------------
# FUNCION 1: Obtener monómero
# ---------------------------------------------------------------------------
def _get_monomer_mol2(polymer_type: str, working_folder: str):
    polymer_type = polymer_type.upper()
    if polymer_type not in POLYMER_DB:
        raise ValueError(f"Polymer {polymer_type} not in database")

    db = POLYMER_DB[polymer_type]
    mol2_filename = POLYMER_MOL2_FILES.get(polymer_type)
    if not mol2_filename:
        raise RuntimeError(f"No mol2 file defined for {polymer_type}")

    mol2_source = os.path.join(POLYMER_MOL2_DIR, mol2_filename)
    if not os.path.exists(mol2_source):
        raise RuntimeError(f"mol2 file not found: {mol2_source}")

    import shutil
    os.makedirs(working_folder, exist_ok=True)
    mol2_out = f"{working_folder}/monomer_gaff2.mol2"
    shutil.copy(mol2_source, mol2_out)

    frcmod_out = f"{working_folder}/monomer.frcmod"
    env = os.environ.copy()
    env["AMBERHOME"] = AMBERHOME
    env["PATH"] = AMBER_BIN + ":" + env.get("PATH", "")

    cmd = (f"{AMBER_BIN}/parmchk2 -i {os.path.basename(mol2_out)} -f mol2 "
           f"-o {os.path.basename(frcmod_out)} -s gaff2")

    result = subprocess.run(cmd, shell=True, cwd=working_folder,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"parmchk2 failed: {result.stderr[-300:]}")

    return mol2_out, frcmod_out


# ---------------------------------------------------------------------------
# FUNCION 2: Construir topología
# ---------------------------------------------------------------------------
def build_polymer_topology(
    polymer_type: str,
    n_chains: int,
    chain_length: int,
    working_folder: str,
    box_size: float = None,
) -> str:
    """
    Construye topología completa con tleap + packmol + cpptraj.
    """
    import shutil as _shutil
    import time as _time

    if os.path.exists(working_folder):
        _shutil.rmtree(working_folder)
    _time.sleep(0.5)
    os.makedirs(working_folder, exist_ok=True)

    polymer_type = polymer_type.upper()
    if polymer_type not in POLYMER_DB:
        return json.dumps({"error": f"Polymer not supported: {polymer_type}"})
    db = POLYMER_DB[polymer_type]

    env = os.environ.copy()
    env["AMBERHOME"] = AMBERHOME
    env["PATH"] = AMBER_BIN + ":" + env.get("PATH", "")

    try:
        mol2_file, frcmod_file = _get_monomer_mol2(polymer_type, working_folder)
        monomer_name = db.get("monomer_name", polymer_type[:3].upper())
        chain_sequence = " ".join([monomer_name] * chain_length)

        # tleap cadena única
        leap_single = f"""source leaprc.gaff2
loadamberparams {os.path.abspath(frcmod_file)}
{monomer_name} = loadmol2 {os.path.abspath(mol2_file)}
chain = sequence {{ {chain_sequence} }}
saveamberparm chain {os.path.abspath(working_folder)}/chain_single.prmtop {os.path.abspath(working_folder)}/chain_single.inpcrd
savepdb chain {os.path.abspath(working_folder)}/chain_single.pdb
quit
"""
        with open(f"{working_folder}/build_chain.leap", "w") as f:
            f.write(leap_single)
        subprocess.run(f"{AMBER_BIN}/tleap -f build_chain.leap",
            shell=True, cwd=working_folder,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

        if not os.path.exists(f"{working_folder}/chain_single.prmtop"):
            return json.dumps({"error": "tleap chain failed"})
        logger.info(f"Single chain topology built: {working_folder}/chain_single.prmtop")

        # tleap caja completa
        chain_defs = ""
        spacing = max(8.0, chain_length * 1.0)
        for i in range(n_chains):
            chain_defs += f"chain{i} = sequence {{ {chain_sequence} }}\n"
            ix, iy, iz = i % 3, (i // 3) % 3, i // 9
            chain_defs += f"translate chain{i} {{ {ix*spacing} {iy*spacing} {iz*spacing} }}\n"

        all_chains = " ".join([f"chain{i}" for i in range(n_chains)])
        prmtop_final = os.path.abspath(f"{working_folder}/polymer_box.prmtop")
        inpcrd_tleap = os.path.abspath(f"{working_folder}/polymer_box_tleap.inpcrd")
        pdb_tleap    = os.path.abspath(f"{working_folder}/chain_all_tleap.pdb")

        leap_box = f"""source leaprc.gaff2
loadamberparams {os.path.abspath(frcmod_file)}
{monomer_name} = loadmol2 {os.path.abspath(mol2_file)}
{chain_defs}box = combine {{ {all_chains} }}
setbox box vdw 5.0
saveamberparm box {prmtop_final} {inpcrd_tleap}
savepdb box {pdb_tleap}
quit
"""
        with open(f"{working_folder}/build_box.leap", "w") as f:
            f.write(leap_box)

        r2 = subprocess.run(f"{AMBER_BIN}/tleap -f build_box.leap",
            shell=True, cwd=working_folder,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

        if not os.path.exists(prmtop_final):
            return json.dumps({"error": f"tleap box failed: {r2.stdout[-400:]}"})
        logger.info(f"Box topology built: {prmtop_final}")

        # Extraer cadena única del PDB para packmol
        with open(pdb_tleap) as f:
            all_pdb_lines = f.readlines()
        chain1_lines = [l for l in all_pdb_lines
                        if l.startswith("ATOM") and int(l[22:26].strip()) <= chain_length]
        chain1_lines.append("END\n")
        single_chain_pdb = f"{working_folder}/chain_single_for_packmol.pdb"
        with open(single_chain_pdb, "w") as f:
            f.writelines(chain1_lines)

        # Calcular box_size por densidad
        if box_size is None:
            mw  = db.get("monomer_mw_gmol", 100.0)
            rho = db.get("density_exp_gcm3", 1.0)
            n_total = n_chains * chain_length
            mass_g = (n_total * mw) / 6.022e23
            vol_A3 = (mass_g / rho) * 1e24
            box_size = (vol_A3 * 2.0) ** (1/3)
            box_size = max(20.0, min(box_size, 80.0))
            logger.info(f"Packmol box: {box_size:.1f} A (target rho={rho})")

        # packmol
        packed_pdb = os.path.abspath(f"{working_folder}/polymer_box_packed.pdb")
        packmol_inp = f"""tolerance 3.5
filetype pdb
output {packed_pdb}
structure {os.path.abspath(single_chain_pdb)}
  number {n_chains}
  inside box 0. 0. 0. {box_size:.1f} {box_size:.1f} {box_size:.1f}
end structure
"""
        with open(f"{working_folder}/packmol_box.inp", "w") as f:
            f.write(packmol_inp)
        r3 = subprocess.run(f"{AMBER_BIN}/packmol < packmol_box.inp",
            shell=True, cwd=working_folder,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

        if not os.path.exists(packed_pdb):
            return json.dumps({"error": f"packmol failed: {r3.stdout[-200:]}"})
        logger.info(f"Polymer box built with packmol: {packed_pdb}")

        # cpptraj
        packed_rst7  = os.path.abspath(f"{working_folder}/polymer_box_packed.rst7")
        inpcrd_final = os.path.abspath(f"{working_folder}/polymer_box.inpcrd")
        cpptraj_inp = f"""parm {prmtop_final}
loadcrd {packed_pdb} name packmol
crdout packmol {packed_rst7} restrt
run
quit
"""
        with open(f"{working_folder}/fix_coords.cpptraj", "w") as f:
            f.write(cpptraj_inp)
        r4 = subprocess.run(f"{AMBER_BIN}/cpptraj -i fix_coords.cpptraj",
            shell=True, cwd=working_folder,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

        if not os.path.exists(packed_rst7):
            return json.dumps({"error": f"cpptraj failed: {r4.stderr[-200:]}"})

        # Añadir dimensiones de caja al rst7
        with open(packed_rst7) as f:
            rst7_content = f.read()
        box_line = f"  {box_size:.7f}  {box_size:.7f}  {box_size:.7f}  90.0000000  90.0000000  90.0000000"
        with open(inpcrd_final, "w") as f:
            f.write(rst7_content.rstrip() + "\n" + box_line + "\n")
        logger.info(f"Coordinates from packmol+cpptraj: {inpcrd_final}")

        import parmed
        parm = parmed.load_file(prmtop_final, inpcrd_final)
        n_atoms    = len(parm.atoms)
        n_residues = len(parm.residues)
        logger.info(f"Topology complete: {n_atoms} atoms, {n_residues} residues")

        return json.dumps({
            "polymer": polymer_type,
            "polymer_name": db["name"],
            "n_chains": n_chains,
            "chain_length": chain_length,
            "n_atoms": n_atoms,
            "n_residues": n_residues,
            "prmtop": prmtop_final,
            "inpcrd": inpcrd_final,
            "box_size": box_size,
        })

    except Exception as e:
        logger.error(f"Error building topology: {e}")
        return json.dumps({"error": str(e), "polymer": polymer_type})



# ===========================================================================
# NÚCLEO MD REESCRITO (revisión septiembre 2026)
# ===========================================================================

# ===========================================================================
# NÚCLEO MD — helpers compartidos (cutoff adaptativo, parsers correctos)
# Insertado por la revisión de septiembre 2026.
# Reglas de integridad:
#   - Todo valor devuelto lleva  "source": "Amber"  y  "method": "MD_*".
#   - Un cálculo que falla se marca  "status": "failed"  y NO se sustituye
#     por ningún valor de tabla/experimental.
# ===========================================================================

def _amber_env():
    """Entorno para pmemd/tleap sin CUDA_VISIBLE_DEVICES (no chocar con Ollama)."""
    import os
    env = os.environ.copy()
    env["AMBERHOME"] = AMBERHOME
    env["PATH"] = AMBER_BIN + ":" + env.get("PATH", "")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _box_radius_from_inpcrd(inpcrd_path: str):
    """Radio de la mayor esfera inscrita en la caja (= min(lado)/2 para caja
    ortorrómbica). pmemd exige cutoff < este radio (regla de imagen mínima).
    Devuelve None si no puede leerse la caja."""
    import os
    try:
        import parmed as pmd
        crd = pmd.load_file(inpcrd_path)
        if crd.box is not None:
            a, b, c = crd.box[0], crd.box[1], crd.box[2]
            return min(a, b, c) / 2.0
    except Exception:
        pass
    # Fallback: leer la última línea del .inpcrd (formato Amber: 3 lados + 3 ángulos)
    try:
        with open(inpcrd_path) as f:
            lines = f.readlines()
        parts = lines[-1].split()
        if len(parts) >= 3:
            vals = [float(x) for x in parts[:3]]
            return min(vals) / 2.0
    except Exception:
        pass
    return None


def _adaptive_cutoff(inpcrd_path: str, desired: float = 8.0) -> float:
    """Cutoff seguro para el tamaño de caja: min(desired, 0.9*radio_esfera).
    Evita el abort 'max pairlist cutoff must be less than unit cell max sphere'."""
    r = _box_radius_from_inpcrd(inpcrd_path)
    if r is None:
        return desired
    safe = 0.9 * r
    cut = min(desired, safe)
    # Suelo razonable: por debajo de 6 Å la física no enlazada se degrada mucho.
    return round(max(cut, 6.0), 2) if safe >= 6.0 else round(safe, 2)


def _parse_min_energy(min_out_path: str):
    """Energía final de una minimización pmemd. Se lee del bloque FINAL RESULTS,
    NO de un inexistente 'FINAL ENERGY' (bug del código anterior)."""
    import os
    if not os.path.exists(min_out_path):
        return None
    try:
        with open(min_out_path) as f:
            lines = f.readlines()
    except Exception:
        return None
    for i, line in enumerate(lines):
        if "FINAL RESULTS" in line:
            # Buscar la fila NSTEP ... y la siguiente con los números
            for j in range(i, min(i + 8, len(lines))):
                if "NSTEP" in lines[j] and "ENERGY" in lines[j]:
                    for k in range(j + 1, min(j + 4, len(lines))):
                        parts = lines[k].split()
                        if len(parts) >= 2:
                            try:
                                return float(parts[1])
                            except ValueError:
                                continue
    return None


def _parse_densities(mdout_path: str):
    """Lista de densidades (g/cm³) de un mdout NPT.
    - Usa regex 'Density = <num>' (formato real verificado en Drago).
    - CORTA al llegar al bloque final 'A V E R A G E S' para no leer el
      frame basura (el parser ingenuo capturaba 0.03 del bloque de cierre)."""
    import os, re
    if not os.path.exists(mdout_path):
        return []
    densities = []
    dens_re = re.compile(r'Density\s*=\s*([\d.]+)')
    with open(mdout_path) as f:
        for line in f:
            # Los promedios/RMS finales de Amber repiten 'Density' con otro
            # significado — parar antes de leerlos.
            if "A V E R A G E S" in line or "R M S" in line:
                break
            m = dens_re.search(line)
            if m:
                try:
                    densities.append(float(m.group(1)))
                except ValueError:
                    pass
    return densities


def _parse_volumes(mdout_path: str):
    """Serie de volumen (Å³) por frame de un mdout NPT. Necesaria para el
    módulo de bulk por fluctuaciones: K = kB*T*<V>/<dV²>.
    Corta en el bloque de promedios finales para no leer el frame basura."""
    import os, re
    if not os.path.exists(mdout_path):
        return []
    volumes = []
    vol_re = re.compile(r'VOLUME\s*=\s*([\d.]+)')
    with open(mdout_path) as f:
        for line in f:
            if "A V E R A G E S" in line or "R M S" in line:
                break
            m = vol_re.search(line)
            if m:
                try:
                    volumes.append(float(m.group(1)))
                except ValueError:
                    pass
    return volumes


def _tail_mean_std(values, tail_frac: float = 0.5):
    """Media y std de la cola equilibrada (último tail_frac de la serie).
    Devuelve (mean, std, n_tail) o (None, None, 0) si no hay datos."""
    import numpy as np
    if not values:
        return None, None, 0
    arr = np.asarray(values, dtype=float)
    start = int(len(arr) * (1.0 - tail_frac))
    tail = arr[start:] if len(arr) - start >= 1 else arr
    return float(tail.mean()), float(tail.std()), int(len(tail))


def _density_converged(densities, block: int = 10, tol: float = 0.02) -> bool:
    """¿La densidad se ha estabilizado? Compara la media de los últimos 'block'
    frames con la de los 'block' anteriores; convergida si el cambio relativo
    < tol. Calibrado con la curva real de PLA (mesetea hacia el frame ~10)."""
    import numpy as np
    if len(densities) < 2 * block:
        return False
    arr = np.asarray(densities, dtype=float)
    recent = arr[-block:].mean()
    previous = arr[-2 * block:-block].mean()
    if previous <= 0:
        return False
    return bool(abs(recent - previous) / previous < tol)


# ===========================================================================
# FUNCIONES MD REESCRITAS (septiembre 2026)
# Sustituyen a run_minimization / run_npt_equilibration / compute_polymer_*
# ===========================================================================

def run_minimization(working_folder: str, prmtop: str, inpcrd: str,
                     max_cycles: int = 2000, cutoff: float = None) -> dict:
    """Minimización con pmemd.cuda. Cutoff adaptativo al tamaño de caja.
    Verifica éxito por existencia del .rst7 (los avisos IEEE de pmemd son
    inofensivos), y lee la energía del bloque FINAL RESULTS."""
    import os, subprocess, json, time
    wf = os.path.abspath(working_folder)
    # Crear el directorio de trabajo si no existe (es un directorio de SALIDA,
    # no un requisito previo). Antes fallaba con 'working_folder not found',
    # lo que hacía imposible calcular densidad en una carpeta nueva.
    os.makedirs(wf, exist_ok=True)

    prmtop = os.path.abspath(prmtop)
    inpcrd = os.path.abspath(inpcrd)
    if cutoff is None:
        cutoff = _adaptive_cutoff(inpcrd, desired=8.0)

    ts = int(time.time() * 1000) % 100000
    min_in  = os.path.join(wf, f"min_{ts}.in")
    min_out = os.path.join(wf, f"min_{ts}.out")
    min_rst = os.path.join(wf, f"min_{ts}.rst7")

    with open(min_in, "w") as f:
        f.write(f"""Minimization
 &cntrl
  imin=1, maxcyc={max_cycles}, ncyc={max_cycles//2},
  ntb=1, cut={cutoff},
  ntpr=100,
 /
""")

    cmd = (f"{AMBER_BIN}/pmemd.cuda -AllowSmallBox -O "
           f"-i {min_in} -o {min_out} -p {prmtop} -c {inpcrd} "
           f"-r {min_rst} -inf {os.path.join(wf,'min.info')}")
    subprocess.run(cmd, shell=True, cwd=wf, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True, env=_amber_env())

    if not os.path.exists(min_rst):
        tail = ""
        if os.path.exists(min_out):
            with open(min_out) as f:
                tail = "".join(f.readlines()[-15:])
        return {"status": "failed",
                "error": f"Minimization produced no rst7 (cut={cutoff}). Tail:\n{tail[-400:]}"}

    energy = _parse_min_energy(min_out)
    logger.info(f"Minimization OK (cut={cutoff}). Final energy: {energy} kcal/mol")
    return {"status": "completed", "final_energy_kcal_mol": energy,
            "rst7": min_rst, "output": min_out, "cutoff": cutoff}


def run_heating(working_folder: str, prmtop: str, rst7_input: str,
                temperature_K: float = 300.0, n_steps: int = 25000,
                cutoff: float = None) -> dict:
    """Calentamiento NVT gradual de 100 K a temperature_K antes del NPT.
    Estabiliza el sistema recién empaquetado."""
    import os, subprocess
    wf = os.path.abspath(working_folder)
    os.makedirs(wf, exist_ok=True)
    prmtop = os.path.abspath(prmtop)
    rst7_input = os.path.abspath(rst7_input)
    if cutoff is None:
        cutoff = _adaptive_cutoff(rst7_input, desired=8.0)

    heat_in  = os.path.join(wf, f"heat_{int(temperature_K)}K.in")
    heat_out = os.path.join(wf, f"heat_{int(temperature_K)}K.out")
    heat_rst = os.path.join(wf, f"heat_{int(temperature_K)}K.rst7")

    with open(heat_in, "w") as f:
        f.write(f"""Heating 100 -> {temperature_K}K
 &cntrl
  imin=0, irest=0, ntx=1,
  nstlim={n_steps}, dt=0.001,
  ntc=2, ntf=2, cut={cutoff},
  ntb=1, ntp=0,
  tempi=100.0, temp0={temperature_K},
  ntt=3, gamma_ln=2.0, ig=-1,
  nmropt=1,
  ntpr=1000, ntwx=0, ntwr={n_steps},
 /
 &wt type='TEMP0', istep1=0, istep2={n_steps},
      value1=100.0, value2={temperature_K} /
 &wt type='END' /
""")

    cmd = (f"{AMBER_BIN}/pmemd.cuda -AllowSmallBox -O "
           f"-i {heat_in} -o {heat_out} -p {prmtop} -c {rst7_input} "
           f"-r {heat_rst} -inf {os.path.join(wf,'heat.info')}")
    subprocess.run(cmd, shell=True, cwd=wf, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True, env=_amber_env())

    if not os.path.exists(heat_rst):
        return {"status": "failed", "error": f"Heating produced no rst7 (cut={cutoff})"}
    return {"status": "completed", "rst7": heat_rst, "output": heat_out, "cutoff": cutoff}


def run_npt_equilibration(working_folder: str, prmtop: str, rst7_input: str,
                          temperature_K: float = 300.0, n_steps: int = 100000,
                          dt: float = 0.002, cutoff: float = None,
                          restart: bool = True) -> dict:
    """Equilibración NPT. Cutoff adaptativo. Éxito = rst7 presente (corrida
    completa). Si el rst7 NO existe => la corrida murió => FALLO (no se
    inventa densidad de un run truncado). Densidad = media de la COLA
    equilibrada, nunca de todos los frames."""
    import os, subprocess
    wf = os.path.abspath(working_folder)
    os.makedirs(wf, exist_ok=True)
    prmtop = os.path.abspath(prmtop)
    rst7_input = os.path.abspath(rst7_input)
    if cutoff is None:
        cutoff = _adaptive_cutoff(rst7_input, desired=8.0)

    npt_in  = os.path.join(wf, f"npt_{int(temperature_K)}K.in")
    npt_out = os.path.join(wf, f"npt_{int(temperature_K)}K.out")
    npt_rst = os.path.join(wf, f"npt_{int(temperature_K)}K.rst7")
    npt_traj = os.path.join(wf, f"npt_{int(temperature_K)}K.nc")

    irest, ntx = (1, 5) if restart else (0, 1)
    with open(npt_in, "w") as f:
        f.write(f"""NPT equilibration at {temperature_K}K
 &cntrl
  imin=0, irest={irest}, ntx={ntx},
  nstlim={n_steps}, dt={dt},
  ntc=2, ntf=2, cut={cutoff},
  ntb=2, ntp=1, barostat=1, taup=2.0,
  temp0={temperature_K}, tempi={temperature_K},
  ntt=3, gamma_ln=2.0, ig=-1,
  ntpr=1000, ntwx=0, ntwr={n_steps}, ioutfm=1,
 /
""")

    cmd = (f"{AMBER_BIN}/pmemd.cuda -AllowSmallBox -O "
           f"-i {npt_in} -o {npt_out} -p {prmtop} -c {rst7_input} "
           f"-r {npt_rst} -x {npt_traj} -inf {os.path.join(wf,'npt.info')}")
    logger.info(f"NPT at {temperature_K}K (cut={cutoff}, {n_steps} steps)...")
    subprocess.run(cmd, shell=True, cwd=wf, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True, env=_amber_env())

    densities = _parse_densities(npt_out)
    volumes = _parse_volumes(npt_out)

    # rst7 ausente = corrida truncada. Es un FALLO, no un resultado.
    if not os.path.exists(npt_rst):
        return {"status": "failed",
                "error": f"NPT truncated (no rst7). Parsed {len(densities)} density frames "
                         f"before dying — treated as FAILURE, not a result.",
                "n_frames": len(densities)}

    mean, std, n_tail = _tail_mean_std(densities, tail_frac=0.5)
    converged = _density_converged(densities)
    if mean is None:
        return {"status": "failed", "error": "NPT completed but no density parsed"}

    logger.info(f"NPT complete. Density (tail mean of {n_tail}): {mean:.4f} "
                f"+/- {std:.4f} g/cm3, converged={converged}")
    return {"status": "completed", "temperature_K": temperature_K,
            "avg_density_gcm3": round(mean, 4), "std_density_gcm3": round(std, 4),
            "n_tail_frames": n_tail, "converged": converged,
            "volumes_A3": volumes,
            "rst7": npt_rst, "output": npt_out, "cutoff": cutoff}


def _equilibrate(polymer_type, working_folder, n_chains, chain_length,
                 temperature_K, topology_folder=None):
    """Pipeline común: obtener/construir topología -> min -> heat -> NPT.
    Devuelve dict con prmtop, rst7 equilibrado, densidad y n_atoms, o
    {'status':'failed', ...}. NO cae a valores experimentales."""
    import os, json
    polymer_type = polymer_type.upper()
    if polymer_type not in POLYMER_DB:
        return {"status": "failed", "error": f"Polymer not supported: {polymer_type}"}
    db = POLYMER_DB[polymer_type]

    # 1) topología
    if topology_folder and os.path.exists(f"{topology_folder}/polymer_box.prmtop"):
        prmtop = os.path.abspath(f"{topology_folder}/polymer_box.prmtop")
        inpcrd = os.path.abspath(f"{topology_folder}/polymer_box.inpcrd")
        try:
            import parmed as pmd
            n_atoms = len(pmd.load_file(prmtop).atoms)
        except Exception:
            n_atoms = 0
        logger.info(f"Reusing topology from {topology_folder}")
    elif db.get("monomer_smiles"):
        from polymer_builder import build_polymer_from_smiles
        topo = build_polymer_from_smiles(
            polymer_name=polymer_type, smiles=db["monomer_smiles"],
            n_chains=n_chains, chain_length=chain_length,
            working_folder=working_folder)
        if "error" in topo:
            return {"status": "failed", "error": f"Topology failed: {topo['error'][:120]}"}
        prmtop, inpcrd = topo["prmtop"], topo["inpcrd"]
        n_atoms = topo.get("n_atoms", 0)
    else:
        topo = json.loads(build_polymer_topology(polymer_type, n_chains, chain_length, working_folder))
        if "error" in topo:
            return {"status": "failed", "error": f"Topology failed: {topo['error'][:120]}"}
        prmtop, inpcrd = topo["prmtop"], topo["inpcrd"]
        n_atoms = topo.get("n_atoms", 0)

    cutoff = _adaptive_cutoff(inpcrd, desired=8.0)
    logger.info(f"Adaptive cutoff for this box: {cutoff} A")

    # 2) minimización
    mn = run_minimization(working_folder, prmtop, inpcrd, cutoff=cutoff)
    if mn["status"] != "completed":
        return {"status": "failed", "error": mn["error"]}

    # 3) calentamiento
    ht = run_heating(working_folder, prmtop, mn["rst7"],
                     temperature_K=temperature_K, cutoff=cutoff)
    if ht["status"] != "completed":
        return {"status": "failed", "error": ht["error"]}

    # 4) NPT (empieza sin velocidades del heat -> restart=True lee ntx=5)
    npt = run_npt_equilibration(working_folder, prmtop, ht["rst7"],
                                temperature_K=temperature_K, n_steps=100000,
                                cutoff=cutoff, restart=True)
    if npt["status"] != "completed":
        return {"status": "failed", "error": npt["error"], "n_frames": npt.get("n_frames")}

    return {"status": "completed", "prmtop": prmtop, "rst7": npt["rst7"],
            "n_atoms": n_atoms, "cutoff": cutoff,
            "density_gcm3": npt["avg_density_gcm3"],
            "density_std": npt["std_density_gcm3"],
            "converged": npt["converged"], "npt_out": npt["output"]}


def compute_polymer_density(polymer_type: str, working_folder: str,
                            n_chains: int = 10, chain_length: int = 20,
                            temperature_K: float = 300.0,
                            n_steps_equil: int = 100000,
                            topology_folder: str = None) -> str:
    """Densidad por MD (Amber/GAFF2). SIEMPRE calculada; si el MD falla,
    devuelve status=failed (NO sustituye por valor experimental)."""
    import json
    eq = _equilibrate(polymer_type, working_folder, n_chains, chain_length,
                      temperature_K, topology_folder)
    if eq["status"] != "completed":
        logger.error(f"Density {polymer_type} FAILED: {eq['error']}")
        return json.dumps({"polymer": polymer_type.upper(), "status": "failed",
                           "property": "density", "error": eq["error"],
                           "source": "Amber", "method": "MD_NPT"})

    db = POLYMER_DB[polymer_type.upper()]
    density = eq["density_gcm3"]
    result = {
        "polymer": polymer_type.upper(), "polymer_name": db["name"],
        "status": "completed", "temperature_K": temperature_K,
        "density_gcm3": density, "density_std_gcm3": eq["density_std"],
        "converged": eq["converged"], "n_chains": n_chains,
        "chain_length": chain_length, "n_atoms": eq["n_atoms"],
        "source": "Amber", "method": "MD_NPT",
        # referencia SOLO para discusión, marcada como tal — nunca rankeable:
        "reference_density_gcm3": db["density_exp_gcm3"],
        "reference_source": "CRC Handbook",
        "error_vs_reference_pct": round(abs(density - db["density_exp_gcm3"])
                                        / db["density_exp_gcm3"] * 100, 2),
    }
    logger.info(f"Density {polymer_type}: {density} g/cm3 (MD) "
                f"vs {db['density_exp_gcm3']} (ref)")
    return json.dumps(result)


def _minimize_energy_at_box(prmtop, base_rst, working_folder, tag,
                            nx, ny, nz, temperature_K, max_cycles=5000):
    """Reescala la caja (nx,ny,nz) con cpptraj, minimiza, y devuelve la
    energía potencial final (kcal/mol) del sistema deformado. None si falla."""
    import os, subprocess
    wf = os.path.abspath(working_folder)
    scaled_rst = os.path.join(wf, f"{tag}_scaled.rst7")
    cpptraj_in = os.path.join(wf, f"{tag}_box.cpptraj")
    with open(cpptraj_in, "w") as f:
        f.write(f"parm {prmtop}\n"
                f"trajin {base_rst}\n"
                f"box x {nx} y {ny} z {nz} alpha 90 beta 90 gamma 90\n"
                f"trajout {scaled_rst} restart\n"
                f"run\nquit\n")
    subprocess.run(f"{AMBER_BIN}/cpptraj -i {os.path.basename(cpptraj_in)}",
                   shell=True, cwd=wf, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True, env=_amber_env())
    if not os.path.exists(scaled_rst):
        return None
    cut_d = _adaptive_cutoff(scaled_rst, desired=8.0)
    min_in  = os.path.join(wf, f"{tag}_min.in")
    min_out = os.path.join(wf, f"{tag}_min.out")
    min_rst = os.path.join(wf, f"{tag}_min.rst7")
    with open(min_in, "w") as f:
        f.write(f"""Minimization deformed {tag}
 &cntrl
  imin=1, maxcyc={max_cycles}, ncyc={max_cycles//2},
  ntb=1, cut={cut_d}, ntpr=100,
 /
""")
    cmd = (f"{AMBER_BIN}/pmemd.cuda -AllowSmallBox -O "
           f"-i {min_in} -o {min_out} -p {prmtop} -c {scaled_rst} "
           f"-r {min_rst} -inf {os.path.join(wf, tag+'_min.info')}")
    subprocess.run(cmd, shell=True, cwd=wf, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True, env=_amber_env())
    if not os.path.exists(min_rst):
        return None
    return _parse_min_energy(min_out)


def compute_polymer_modulus(polymer_type: str, working_folder: str,
                            n_chains: int = 10, chain_length: int = 20,
                            temperature_K: float = 300.0,
                            strains=None, poisson_ratio: float = 0.40,
                            topology_folder: str = None) -> str:
    """Módulo de Young REAL por deformación hidrostática + energía (Opción C).

    Método (DETERMINISTA, sin ruido estadístico):
      1. Equilibrar el sistema (min->heat->NPT) a densidad física.
      2. Para cada deformación hidrostática eps, reescalar la caja a
         L*(1+eps) en los 3 ejes, MINIMIZAR, y leer U(eps).
      3. Ajustar parábola U(eps) = a*eps^2 + b*eps + c.
         Curvatura d2U/deps2 = 2a = 9*K*V0  =>  K = 2a / (9*V0).
      4. E = 3*K*(1 - 2*nu), con nu DECLARADO explícitamente.

    Ventaja sobre fluctuaciones: la energía minimizada es limpia y
    reproducible (no depende de <dV^2>), asi que converge sin NPT largos.
    Honestidad: K medido, E derivado con nu visible. source="Amber",
    method="MD_energy_deformation". Fallo de equilibrado => status=failed."""
    import os, json
    import numpy as np

    if strains is None:
        strains = [-0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02]

    eq = _equilibrate(polymer_type, working_folder, n_chains, chain_length,
                      temperature_K, topology_folder)
    if eq["status"] != "completed":
        logger.error(f"Modulus {polymer_type} FAILED at equilibration: {eq['error']}")
        return json.dumps({"polymer": polymer_type.upper(), "status": "failed",
                           "property": "modulus", "error": eq["error"],
                           "source": "Amber", "method": "MD_energy_deformation"})

    wf = os.path.abspath(working_folder)
    prmtop = eq["prmtop"]
    base_rst = eq["rst7"]

    # Caja equilibrada
    try:
        import parmed as pmd
        crd = pmd.load_file(prmtop, base_rst)
        bx, by, bz = float(crd.box[0]), float(crd.box[1]), float(crd.box[2])
    except Exception as e:
        return json.dumps({"polymer": polymer_type.upper(), "status": "failed",
                           "property": "modulus", "error": f"Cannot read box: {e}",
                           "source": "Amber", "method": "MD_energy_deformation"})
    V0 = bx * by * bz

    # Medir U en eps=0 y en cada deformación hidrostática
    points = []  # (eps, U_kcal)
    all_eps = [0.0] + list(strains)
    for eps in sorted(set(all_eps)):
        tag = f"hydro_{eps:+.4f}".replace(".", "p")
        s = (1.0 + eps)
        U = _minimize_energy_at_box(prmtop, base_rst, wf, tag,
                                    bx*s, by*s, bz*s, temperature_K)
        if U is not None:
            points.append((eps, U))
        else:
            logger.warning(f"Modulus: deformation eps={eps} failed (no energy)")

    if len(points) < 4:
        return json.dumps({"polymer": polymer_type.upper(), "status": "failed",
                           "property": "modulus",
                           "error": f"Only {len(points)} energy points; need >=4",
                           "source": "Amber", "method": "MD_energy_deformation"})

    eps_arr = np.array([p[0] for p in points])
    U_arr = np.array([p[1] for p in points])

    # Ajuste parabólico U = a*eps^2 + b*eps + c
    a, b, c = np.polyfit(eps_arr, U_arr, 2)
    # R^2
    pred = a*eps_arr**2 + b*eps_arr + c
    ss_res = float(np.sum((U_arr - pred)**2))
    ss_tot = float(np.sum((U_arr - U_arr.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

    # Curvatura d2U/deps2 = 2a (kcal/mol). K = 2a/(9*V0) en kcal/mol/A^3.
    # Convertir a GPa: 1 GPa·A^3 = 0.143932 kcal/mol => 1 kcal/mol/A^3 = 1/0.143932 GPa
    GPA_A3_PER_KCALMOL = 1.0 / 0.143932
    curvature = 2.0 * a
    if curvature <= 0:
        return json.dumps({"polymer": polymer_type.upper(), "status": "failed",
                           "property": "modulus",
                           "error": f"Non-positive energy curvature ({curvature:.3f}); "
                                    f"system not at a stable minimum",
                           "source": "Amber", "method": "MD_energy_deformation",
                           "fit_R2": round(r2, 4)})
    K_GPa = (curvature / (9.0 * V0)) * GPA_A3_PER_KCALMOL

    nu = poisson_ratio
    E_GPa = 3.0 * K_GPa * (1.0 - 2.0*nu)
    converged = bool(r2 > 0.95)  # parábola limpia => fit fiable

    db = POLYMER_DB[polymer_type.upper()]
    result = {
        "polymer": polymer_type.upper(), "polymer_name": db["name"],
        "status": "completed", "temperature_K": temperature_K,
        "K_GPa": round(float(K_GPa), 3),
        "E_GPa": round(float(E_GPa), 3),
        "assumed_poisson_ratio": nu,
        "fit_R2": round(r2, 4), "converged": converged,
        "n_strain_points": len(points),
        "equilibrated_density_gcm3": eq["density_gcm3"],
        "n_chains": n_chains, "chain_length": chain_length, "n_atoms": eq["n_atoms"],
        "source": "Amber", "method": "MD_energy_deformation",
        "reference_E_GPa": db["E_exp_GPa"], "reference_source": "CRC Handbook",
        "error_vs_reference_pct": round(abs(E_GPa - db["E_exp_GPa"])
                                        / db["E_exp_GPa"] * 100, 2),
    }
    logger.info(f"Modulus {polymer_type}: K={K_GPa:.2f} GPa, E={E_GPa:.2f} GPa "
                f"(nu={nu}, R2={r2:.3f}, converged={converged}) vs {db['E_exp_GPa']} (ref)")
    return json.dumps(result)


def get_polymer_reference(polymer_type: str) -> dict:
    """Devuelve los valores experimentales de referencia para un polímero."""
    polymer_type = polymer_type.upper()
    if polymer_type not in POLYMER_DB:
        return json.dumps({"error": f"Polymer not found: {polymer_type}",
                          "available": list(POLYMER_DB.keys())})
    db = POLYMER_DB[polymer_type]
    return json.dumps({
        "polymer": polymer_type,
        "name": db["name"],
        "Tg_experimental_K": db["Tg_exp_K"],
        "Tg_experimental_C": db["Tg_exp_K"] - 273.15,
        "density_experimental_gcm3": db["density_exp_gcm3"],
        "E_experimental_GPa": db["E_exp_GPa"],
        "source": "CRC Handbook / Polymer Handbook",
    })


# ---------------------------------------------------------------------------
# FALLBACK: Densidad directa de referencia (sin simulación)
# ---------------------------------------------------------------------------


# ===========================================================================
# Tg — NO IMPLEMENTADO como cálculo real todavía.
# Regla de integridad: NO devolvemos el valor experimental disfrazado de
# calculado (eso era el stub anterior). Devolvemos un fallo honesto hasta
# implementar la rampa de enfriamiento MD real.
# ===========================================================================
def compute_glass_transition(polymer_type: str, working_folder: str,
                             n_chains: int = 10, chain_length: int = 20,
                             T_high_K: float = 600.0, T_low_K: float = 100.0,
                             n_T_steps: int = 10, n_steps_per_T: int = 200000,
                             **kwargs) -> str:
    """Tg por MD (rampa de enfriamiento) — NO IMPLEMENTADO todavía.
    Devuelve status=not_implemented en lugar de un valor experimental
    disfrazado de calculado."""
    import json
    logger.warning(f"compute_glass_transition({polymer_type}): not implemented — "
                   f"returning honest failure, NOT the experimental value")
    return json.dumps({
        "polymer": polymer_type.upper(), "status": "not_implemented",
        "property": "Tg",
        "error": "Tg via MD cooling ramp is not implemented yet. "
                 "Use get_polymer_reference for an experimental value (discussion only).",
        "source": "Amber", "method": "MD_cooling_ramp",
    })
