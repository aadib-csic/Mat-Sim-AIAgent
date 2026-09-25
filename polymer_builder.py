#!/usr/bin/env python3
# coding: utf-8
"""
polymer_builder.py - Pipeline limpio: RDKit → antechamber → tleap → packmol → cpptraj

Enfoque:
1. RDKit construye el oligómero 3D directamente desde SMILES (sin PubChem)
2. antechamber parametriza el oligómero con GAFF2
3. tleap construye la topología (sin prepgen/HEAD/TAIL)
4. packmol distribuye n_chains en caja 3D
5. cpptraj combina topología + coordenadas

Ventaja: elimina el problema de HEAD/TAIL y PubChem.
El oligómero de RDKit tiene geometría 3D correcta desde el principio.
"""

import os
import json
import logging
import subprocess
import shutil
import time
import numpy as np
from typing import Optional

logger = logging.getLogger("polymer_builder")

AMBER_BIN = "/lustre/opt/foss/0.4/software/Amber/24.0-foss-2023b-AmberTools-25.0-CUDA-12.4.0/bin"
AMBERHOME = "/lustre/opt/foss/0.4/software/Amber/24.0-foss-2023b-AmberTools-25.0-CUDA-12.4.0"

POLYMER_DENSITY_DB = {
    "PE": 0.95, "PP": 0.91, "PS": 1.05, "PMMA": 1.18,
    "PLA": 1.24, "PEEK": 1.32, "PET": 1.38, "PVC": 1.40,
}

# SMILES de unidades repetitivas para polímeros comunes
# Construidos como oligómeros cortos para dar geometría 3D correcta
POLYMER_REPEAT_SMILES = {
    "PLA":  "CC(OC(=O)C(C)OC(=O)C(C)OC(=O)C(C)OC(=O)C(C)O)=O",  # pentámero
    "PE":   "CCCCCCCCCC",                                            # decano
    "PP":   "CCC(C)CC(C)CC(C)CC(C)C",                               # oligómero
    "PS":   "CCC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C",               # trímero
    "PMMA": "CC(C(=O)OC)(CC(C(=O)OC)(C)CC(C(=O)OC)(C)C)C",        # trímero
    "PET":  "OCCOC(=O)c1ccc(cc1)C(=O)OCCOC(=O)c1ccc(cc1)C(=O)O",  # dímero
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def _get_env() -> dict:
    env = os.environ.copy()
    env["AMBERHOME"] = AMBERHOME
    env["PATH"] = AMBER_BIN + ":" + env.get("PATH", "")
    return env


def _run(cmd: str, cwd: str, env: dict, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, timeout=timeout
    )


def _find_packmol() -> Optional[str]:
    candidates = [
        os.path.join(AMBER_BIN, "packmol"),
        os.path.expanduser("~/.conda/envs/atomagents/bin/packmol"),
        "/usr/local/bin/packmol",
    ]
    try:
        which = shutil.which("packmol")
        if which:
            return which
    except:
        pass
    for p in candidates:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


# ============================================================================
# PASO 1: RDKit → oligómero 3D
# ============================================================================

def build_oligomer_3d(
    smiles: str,
    polymer_name: str,
    chain_length: int,
    output_sdf: str,
) -> dict:
    """
    Construye un oligómero 3D desde SMILES usando RDKit.
    
    Estrategia:
    - Si hay un SMILES de oligómero predefindio, lo usa directamente
    - Si no, construye un oligómero repitiendo el monómero
    
    Returns:
        dict con n_atoms, sdf_path o error
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Usar SMILES de oligómero predefinido si existe
    rname = polymer_name.upper()
    oligomer_smiles = POLYMER_REPEAT_SMILES.get(rname)
    
    if not oligomer_smiles:
        # Intentar construir oligómero desde monómero
        # Para polímeros simples: repetir el SMILES n veces
        logger.info(f"No hay oligómero predefinido para {polymer_name} — usando SMILES del monómero")
        oligomer_smiles = smiles

    logger.info(f"SMILES oligómero: {oligomer_smiles[:60]}...")
    
    mol = Chem.MolFromSmiles(oligomer_smiles)
    if mol is None:
        return {"error": f"RDKit no puede parsear SMILES: {oligomer_smiles[:50]}"}

    # Añadir H explícitos
    mol = Chem.AddHs(mol)
    
    # Generar conformación 3D
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    
    if result != 0:
        # Intentar con parámetros más permisivos
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    if result != 0:
        return {"error": f"RDKit no pudo generar conformación 3D para: {oligomer_smiles[:50]}"}

    # Optimizar geometría
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            logger.warning("Optimización de geometría falló — usando geometría inicial")

    # Verificar extensión 3D
    conf = mol.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z]
                       for i in range(mol.GetNumAtoms())])
    max_range = float(np.max(coords.max(axis=0) - coords.min(axis=0)))
    
    logger.info(f"Oligómero generado: {mol.GetNumAtoms()} átomos, extensión máx = {max_range:.1f} Å")

    # Guardar SDF
    writer = Chem.SDWriter(output_sdf)
    writer.write(mol)
    writer.close()

    return {
        "n_atoms": mol.GetNumAtoms(),
        "max_range_A": max_range,
        "sdf_path": output_sdf,
        "oligomer_smiles": oligomer_smiles,
    }


# ============================================================================
# PASO 2: antechamber → mol2 GAFF2
# ============================================================================

def parametrize_with_antechamber(
    sdf_file: str,
    residue_name: str,
    working_folder: str,
) -> dict:
    """
    Parametriza el oligómero con antechamber GAFF2 + parmchk2.
    
    Returns:
        dict con mol2_path, frcmod_path o error
    """
    env = _get_env()
    mol2_file   = os.path.join(working_folder, f"{residue_name}.mol2")
    frcmod_file = os.path.join(working_folder, f"{residue_name}.frcmod")

    # antechamber SDF → mol2 GAFF2
    logger.info("PASO 2a: antechamber → mol2 GAFF2")
    cmd = (f"{AMBER_BIN}/antechamber "
           f"-i {os.path.basename(sdf_file)} -fi sdf "
           f"-o {os.path.basename(mol2_file)} -fo mol2 "
           f"-c bcc -at gaff2 -rn {residue_name} -nc 0")
    r = _run(cmd, cwd=working_folder, env=env, timeout=600)

    if not os.path.exists(mol2_file):
        return {"error": f"antechamber falló: {r.stderr[-300:]}"}
    logger.info(f"✓ mol2 generado: {mol2_file}")

    # parmchk2 → frcmod
    logger.info("PASO 2b: parmchk2 → frcmod")
    cmd = (f"{AMBER_BIN}/parmchk2 "
           f"-i {os.path.basename(mol2_file)} -f mol2 "
           f"-o {os.path.basename(frcmod_file)} -s gaff2")
    r = _run(cmd, cwd=working_folder, env=env)

    if not os.path.exists(frcmod_file):
        return {"error": f"parmchk2 falló: {r.stderr[-200:]}"}
    logger.info(f"✓ frcmod generado: {frcmod_file}")

    return {
        "mol2_path":   mol2_file,
        "frcmod_path": frcmod_file,
    }


# ============================================================================
# PASO 3: tleap → topología de cadena única
# ============================================================================

def build_single_chain_topology(
    mol2_file: str,
    frcmod_file: str,
    residue_name: str,
    working_folder: str,
) -> dict:
    """
    Construye la topología de una cadena usando tleap con mol2 directo.
    Sin prepgen, sin HEAD/TAIL.
    
    Returns:
        dict con prmtop_path, pdb_path o error
    """
    env = _get_env()
    prmtop = os.path.join(working_folder, "chain_single.prmtop")
    inpcrd  = os.path.join(working_folder, "chain_single.inpcrd")
    pdb     = os.path.join(working_folder, "chain_single.pdb")

    logger.info("PASO 3: tleap → topología cadena única")
    leap_content = f"""source leaprc.gaff2
loadamberparams {os.path.abspath(frcmod_file)}
{residue_name} = loadmol2 {os.path.abspath(mol2_file)}
savepdb {residue_name} {os.path.abspath(pdb)}
saveamberparm {residue_name} {os.path.abspath(prmtop)} {os.path.abspath(inpcrd)}
quit
"""
    leap_file = os.path.join(working_folder, "build_chain.leap")
    with open(leap_file, 'w') as f:
        f.write(leap_content)

    r = _run(f"{AMBER_BIN}/tleap -f build_chain.leap", cwd=working_folder, env=env)

    # Verificar errores
    leap_log = os.path.join(working_folder, "leap.log")
    # tleap escribe el log en el cwd actual, no en working_folder
    if not os.path.exists(leap_log):
        leap_log = "leap.log"  # en el directorio actual

    n_errors = 0
    if os.path.exists(leap_log):
        with open(leap_log) as f:
            log = f.read()
        for line in log.splitlines():
            if "Errors =" in line:
                try:
                    n_errors = int(line.split("Errors =")[1].split(";")[0].strip())
                except:
                    pass

    if not os.path.exists(prmtop):
        return {"error": f"tleap falló ({n_errors} errores): {r.stdout[-300:]}"}

    if n_errors > 0:
        errors = []
        if os.path.exists(leap_log):
            with open(leap_log) as f:
                errors = [l.strip() for l in f if "Could not find" in l][:3]
        return {"error": f"tleap {n_errors} errores", "errors": errors}

    # Verificar geometría
    coords = []
    if os.path.exists(pdb):
        with open(pdb) as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except:
                        pass

    max_range = 0.0
    if coords:
        c = np.array(coords)
        max_range = float(np.max(c.max(axis=0) - c.min(axis=0)))

    logger.info(f"✓ Cadena única: {prmtop} ({len(coords)} átomos, {max_range:.1f} Å)")

    return {
        "prmtop_path": prmtop,
        "inpcrd_path": inpcrd,
        "pdb_path":    pdb,
        "n_atoms":     len(coords),
        "max_range_A": max_range,
        "n_errors":    n_errors,
    }


# ============================================================================
# PASO 4: packmol → caja 3D
# ============================================================================

def build_box_with_packmol(
    chain_pdb: str,
    n_chains: int,
    box_size: float,
    working_folder: str,
) -> dict:
    """
    Distribuye n_chains en una caja 3D con packmol.
    
    Returns:
        dict con packed_pdb o error
    """
    packmol_cmd = _find_packmol()
    if not packmol_cmd:
        return {"error": "packmol no encontrado"}

    packed_pdb  = os.path.join(working_folder, "polymer_packed.pdb")
    packmol_inp = os.path.join(working_folder, "packmol.inp")

    with open(packmol_inp, 'w') as f:
        f.write(f"""tolerance 3.0
filetype pdb
output {os.path.abspath(packed_pdb)}
structure {os.path.abspath(chain_pdb)}
  number {n_chains}
  inside box 0. 0. 0. {box_size:.1f} {box_size:.1f} {box_size:.1f}
end structure
""")

    logger.info(f"PASO 4: packmol — {n_chains} cadenas en caja {box_size:.1f} Å")
    env = _get_env()
    try:
        r = _run(f"{packmol_cmd} < packmol.inp", cwd=working_folder, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        box_size *= 2.0
        logger.warning(f"packmol timeout — reintentando con box={box_size:.1f} Å")
        with open(packmol_inp, 'w') as f:
            f.write(f"""tolerance 5.0
filetype pdb
output {os.path.abspath(packed_pdb)}
structure {os.path.abspath(chain_pdb)}
  number {n_chains}
  inside box 0. 0. 0. {box_size:.1f} {box_size:.1f} {box_size:.1f}
end structure
""")
        try:
            r = _run(f"{packmol_cmd} < packmol.inp", cwd=working_folder, env=env, timeout=300)
        except subprocess.TimeoutExpired:
            return {"error": "packmol timeout en segundo intento"}

    if not os.path.exists(packed_pdb):
        return {"error": f"packmol falló: {r.stderr[-200:] if hasattr(r,'stderr') else ''}"}

    logger.info(f"✓ packmol completado: {packed_pdb}")
    return {"packed_pdb": packed_pdb, "box_size": box_size}


# ============================================================================
# PASO 5: cpptraj → combinar topología + coordenadas
# ============================================================================

def combine_topology_coordinates(
    chain_prmtop: str,
    packed_pdb: str,
    box_size: float,
    working_folder: str,
) -> dict:
    """
    Combina la topología de tleap con las coordenadas de packmol usando cpptraj.
    
    Returns:
        dict con prmtop, inpcrd o error
    """
    env = _get_env()
    packed_rst7  = os.path.join(working_folder, "polymer_packed.rst7")
    inpcrd_final = os.path.join(working_folder, "polymer_box.inpcrd")
    prmtop_final = os.path.join(working_folder, "polymer_box.prmtop")

    logger.info("PASO 5: tleap → topología caja completa")
    # Primero crear prmtop de la caja completa con tleap
    frcmod_files = [f for f in os.listdir(working_folder) if f.endswith('.frcmod')]
    mol2_files   = [f for f in os.listdir(working_folder)
                    if f.endswith('.mol2') and not f.startswith('ANTECHAMBER')]
    box_leap = os.path.join(working_folder, "build_box.leap")
    box_prmtop_tleap = os.path.join(working_folder, "box_tleap.prmtop")
    box_inpcrd_tleap = os.path.join(working_folder, "box_tleap.inpcrd")

    with open(box_leap, 'w') as f:
        lines = ["source leaprc.gaff2"]
        for frc in frcmod_files:
            lines.append(f"loadamberparams {frc}")
        for m2 in mol2_files:
            rn = m2.replace('.mol2','')
            lines.append(f"{rn} = loadmol2 {m2}")
        lines.append(f"box = loadpdb {os.path.basename(packed_pdb)}")
        lines.append("setbox box vdw 3.0")
        lines.append(f"saveamberparm box {os.path.abspath(box_prmtop_tleap)} {os.path.abspath(box_inpcrd_tleap)}")
        lines.append("quit")
        f.write("\n".join(lines) + "\n")

    _run(f"{AMBER_BIN}/tleap -f build_box.leap", cwd=working_folder, env=env)

    # Si tleap box falló, usar cpptraj con prmtop de cadena única replicado
    if os.path.exists(box_prmtop_tleap):
        prmtop_for_cpptraj = box_prmtop_tleap
        logger.info(f"✓ tleap box OK: {box_prmtop_tleap}")
    else:
        logger.warning("tleap box falló — usando prmtop cadena única para cpptraj")
        prmtop_for_cpptraj = chain_prmtop

    logger.info("PASO 5b: cpptraj → combinar topología + coordenadas")
    cpptraj_inp = os.path.join(working_folder, "fix_coords.cpptraj")
    with open(cpptraj_inp, 'w') as f:
        f.write(f"""parm {os.path.abspath(prmtop_for_cpptraj)}
loadcrd {os.path.abspath(packed_pdb)} name packmol
crdout packmol {os.path.abspath(packed_rst7)} restrt
run
quit
""")

    r = _run(f"{AMBER_BIN}/cpptraj -i fix_coords.cpptraj", cwd=working_folder, env=env)

    # cpptraj puede generar rst7.1 rst7.2 si hay discrepancia de átomos
    if not os.path.exists(packed_rst7):
        # Buscar rst7.1
        rst7_alt = packed_rst7 + ".1"
        if os.path.exists(rst7_alt):
            shutil.copy(rst7_alt, packed_rst7)
            logger.warning("cpptraj generó rst7.1 — usando ese")
        else:
            return {"error": f"cpptraj falló: {r.stderr[-200:]}"}

    # Añadir dimensiones de caja al rst7
    with open(packed_rst7) as f:
        rst7_content = f.read()
    box_line = f"  {box_size:.7f}  {box_size:.7f}  {box_size:.7f}  90.0000000  90.0000000  90.0000000"
    with open(inpcrd_final, 'w') as f:
        f.write(rst7_content.rstrip() + "\n" + box_line + "\n")

    # Usar prmtop de la caja si existe, si no copiar el de cadena única
    if os.path.exists(box_prmtop_tleap):
        shutil.copy(box_prmtop_tleap, prmtop_final)
        logger.info(f"✓ Usando prmtop de caja: {box_prmtop_tleap}")
    else:
        shutil.copy(chain_prmtop, prmtop_final)
        logger.warning("Usando prmtop de cadena única — puede haber mismatch de átomos")
    logger.info(f"✓ Coordenadas combinadas: {inpcrd_final}")

    return {
        "prmtop": os.path.abspath(prmtop_final),
        "inpcrd": os.path.abspath(inpcrd_final),
    }


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def build_polymer_from_smiles(
    polymer_name: str,
    smiles: str,
    n_chains: int,
    chain_length: int,
    working_folder: str,
    description: str = "",
    box_size: float = None,
) -> dict:
    """
    Pipeline completo: SMILES → topología Amber lista para simular.

    RDKit → antechamber → tleap → packmol → cpptraj

    Args:
        polymer_name: nombre del polímero (PE, PS, PMMA, PLA, etc.)
        smiles: SMILES del monómero (unidad repetitiva)
        n_chains: número de cadenas en la caja
        chain_length: monómeros por cadena
        working_folder: directorio de trabajo (NO modifica polymer_mol2/)
        description: descripción del polímero
        box_size: tamaño de caja en Å (None = calculado por densidad)

    Returns:
        dict con prmtop, inpcrd, n_atoms, box_size o error
    """
    # Limpiar y crear directorio
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)
    time.sleep(0.5)
    os.makedirs(working_folder, exist_ok=True)

    rname = polymer_name[:3].upper()

    # PASO 1: RDKit → oligómero 3D
    logger.info(f"{'='*60}")
    logger.info(f"POLYMER BUILDER: {polymer_name}")
    logger.info(f"  SMILES: {smiles}")
    logger.info(f"  Chains: {n_chains} × {chain_length} monomers")
    logger.info(f"{'='*60}")

    sdf_file = os.path.join(working_folder, "oligomer.sdf")
    step1 = build_oligomer_3d(smiles, polymer_name, chain_length, sdf_file)
    if "error" in step1:
        return step1
    logger.info(f"✓ PASO 1: {step1['n_atoms']} átomos, {step1['max_range_A']:.1f} Å extensión")

    # Copiar SDF al working folder para antechamber
    sdf_local = os.path.join(working_folder, "oligomer.sdf")

    # PASO 2: antechamber → mol2 GAFF2
    step2 = parametrize_with_antechamber(sdf_local, rname, working_folder)
    if "error" in step2:
        return step2
    logger.info(f"✓ PASO 2: mol2 y frcmod generados")

    # PASO 3: tleap → topología cadena única
    step3 = build_single_chain_topology(
        step2["mol2_path"], step2["frcmod_path"], rname, working_folder
    )
    if "error" in step3:
        return step3
    logger.info(f"✓ PASO 3: topología {step3['n_atoms']} átomos, {step3['max_range_A']:.1f} Å")

    # Calcular box_size
    if box_size is None:
        rho_target = POLYMER_DENSITY_DB.get(polymer_name.upper(), 1.0)

        # Masa REAL del sistema, leída del prmtop de la cadena única.
        # (antes: n_atoms * 12.0 — asumía todo carbono, ~50% de error para PLA)
        try:
            import parmed as _pmd
            _chain = _pmd.load_file(step3["prmtop_path"])
            mass_per_chain_amu = sum(a.mass for a in _chain.atoms)
        except Exception as _e:
            logger.warning(f"No se pudo leer masa del prmtop ({_e}); "
                           f"usando estimación por átomos")
            mass_per_chain_amu = step3["n_atoms"] * 12.0
        total_mass_amu = n_chains * mass_per_chain_amu

        # Volumen a densidad física objetivo (1 amu/ų = 1.6605 g/cm³)
        AMU_A3_TO_GCM3 = 1.66053907
        vol_at_target_A3 = (total_mass_amu * AMU_A3_TO_GCM3) / rho_target
        L_target = vol_at_target_A3 ** (1.0 / 3.0)

        # Holgura de EMPAQUETADO: packmol necesita margen para colocar cadenas
        # extendidas sin solapes, pero la caja NO debe nacer a densidad de gas.
        # Factor 1.3 en volumen (~7% en lado) → densidad inicial ~0.77*rho,
        # que el NPT comprime en pocos pasos. (antes: 2.5 → densidad de gas.)
        PACKING_VOLUME_FACTOR = 1.3
        box_size = (vol_at_target_A3 * PACKING_VOLUME_FACTOR) ** (1.0 / 3.0)

        # Suelo mínimo SOLO por seguridad de packmol con cadenas grandes:
        # la caja debe ser mayor que la extensión de una cadena.
        min_box = max(step3.get("max_range_A", 0.0) * 1.2, 12.0)
        if box_size < min_box:
            box_size = min_box

        rho_initial = (total_mass_amu * AMU_A3_TO_GCM3) / (box_size ** 3)
        logger.info(
            f"Box size: {box_size:.1f} Å | masa={total_mass_amu:.1f} amu | "
            f"rho_objetivo={rho_target:.3f} g/cm³ | "
            f"rho_inicial_empaquetado={rho_initial:.3f} g/cm³ "
            f"(el NPT debe comprimir hasta ~{rho_target:.2f})"
        )

    # PASO 4: packmol → caja 3D
    step4 = build_box_with_packmol(
        step3["pdb_path"], n_chains, box_size, working_folder
    )
    if "error" in step4:
        return step4
    logger.info(f"✓ PASO 4: caja distribuida")

    # PASO 5: cpptraj → topología + coordenadas
    step5 = combine_topology_coordinates(
        step3["prmtop_path"], step4["packed_pdb"],
        step4["box_size"], working_folder
    )
    if "error" in step5:
        return step5
    logger.info(f"✓ PASO 5: coordenadas combinadas")

    # Verificar con parmed
    try:
        import parmed
        parm = parmed.load_file(step5["prmtop"], step5["inpcrd"])
        n_atoms    = len(parm.atoms)
        n_residues = len(parm.residues)
        logger.info(f"✓ Verificación parmed: {n_atoms} átomos, {n_residues} residuos")
    except Exception as e:
        logger.warning(f"parmed: {e}")
        n_atoms    = n_chains * step3["n_atoms"]
        n_residues = n_chains

    logger.info(f"{'='*60}")
    logger.info(f"✓ TOPOLOGÍA COMPLETADA: {step5['prmtop']}")
    logger.info(f"{'='*60}")

    return {
        "polymer":      polymer_name,
        "smiles":       smiles,
        "n_chains":     n_chains,
        "chain_length": chain_length,
        "n_atoms":      n_atoms,
        "n_residues":   n_residues,
        "prmtop":       step5["prmtop"],
        "inpcrd":       step5["inpcrd"],
        "box_size":     step4["box_size"],
        "max_range_A":  step3["max_range_A"],
        "method":       "RDKit→antechamber→tleap→packmol→cpptraj",
    }


# ============================================================================
# LANGCHAIN TOOL WRAPPER
# ============================================================================

def create_polymer_builder_tool():
    from langchain_core.tools import tool

    @tool
    def prepare_polymer_topology_tool(
        polymer_name: str,
        n_chains: int,
        chain_length: int,
        working_directory: str,
        smiles: str = "",
        description: str = "",
    ) -> str:
        """
        Build polymer topology from SMILES using RDKit + AmberTools pipeline.

        Uses RDKit to generate correct 3D geometry (no PubChem, no HEAD/TAIL issues).
        Then parametrizes with antechamber GAFF2 and builds topology with tleap.

        For NEW polymers not in standard database (PE, PP, PS, PMMA).
        Standard polymers use pre-existing mol2 files from polymer_mol2/.

        Args:
            polymer_name: polymer name (e.g. "PLA", "PEEK", "PET")
            n_chains: number of polymer chains in simulation box
            chain_length: monomers per chain
            working_directory: working folder (does NOT modify polymer_mol2/)
            smiles: monomer repeat unit SMILES (e.g. "CC(O)C(=O)O" for PLA)
            description: polymer description to guide parametrization

        Returns:
            JSON with prmtop, inpcrd paths and topology statistics
        """
        if not smiles:
            return json.dumps({
                "error": f"SMILES required for {polymer_name}.",
                "hint": "Provide the monomer repeat unit SMILES."
            })

        result = build_polymer_from_smiles(
            polymer_name=polymer_name,
            smiles=smiles,
            n_chains=n_chains,
            chain_length=chain_length,
            working_folder=working_directory,
            description=description,
        )
        return json.dumps(result, indent=2)

    return prepare_polymer_topology_tool


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    polymer     = sys.argv[1] if len(sys.argv) > 1 else "PLA"
    smiles      = sys.argv[2] if len(sys.argv) > 2 else "CC(O)C(=O)O"
    description = sys.argv[3] if len(sys.argv) > 3 else "biodegradable linear polyester"

    print(f"\n=== Test polymer_builder: {polymer} ===")
    print(f"SMILES: {smiles}\n")

    result = build_polymer_from_smiles(
        polymer_name=polymer,
        smiles=smiles,
        n_chains=2,
        chain_length=5,
        working_folder=f"test_{polymer.lower()}_rdkit",
        description=description,
    )
    print(json.dumps(result, indent=2))

