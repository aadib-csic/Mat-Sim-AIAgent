#!/usr/bin/env python3
# coding: utf-8
"""
reference_data.py — FUENTE ÚNICA de valores de referencia curados.

Consolida las referencias que antes estaban dispersas y duplicadas en
POLYMER_DB (amber_functions) y FALLBACK_REFERENCE (rag_tool).

REGLA DE INTEGRIDAD:
  - Estos son valores EXPERIMENTALES / de literatura, con su fuente.
  - Se usan SOLO en la fase de EVALUACIÓN (comparar cálculo vs referencia).
  - NUNCA entran en el CSV de resultados como si fueran calculados.
  - Toda consulta devuelve el valor CON su 'source' para trazabilidad.

Fuentes:
  - Metales: Mishin et al., Phys. Rev. B 59, 3393 (1999) [potenciales EAM];
    densidades y constantes de red de CRC Handbook.
  - Polímeros: CRC Handbook of Chemistry and Physics; valores típicos de
    grado comercial a temperatura ambiente.
"""

# ---------------------------------------------------------------------------
# METALES
# ---------------------------------------------------------------------------
METAL_REFERENCE = {
    "Al": {
        "name": "Aluminum", "structure": "fcc",
        "molar_mass_g_mol": 26.98,
        "lattice_constant_A": 4.05,
        "density_gcm3": 2.70,
        "C11_GPa": 113.8, "C12_GPa": 61.6, "C44_GPa": 31.6,
        "E_GPa": 70.0, "bulk_modulus_GPa": 76.0, "poisson_ratio": 0.35,
        "source": "Mishin et al. PRB 1999; CRC Handbook",
    },
    "Ni": {
        "name": "Nickel", "structure": "fcc",
        "molar_mass_g_mol": 58.69,
        "lattice_constant_A": 3.52,
        "density_gcm3": 8.91,
        "C11_GPa": 247.0, "C12_GPa": 147.0, "C44_GPa": 125.0,
        "E_GPa": 200.0, "bulk_modulus_GPa": 180.0, "poisson_ratio": 0.31,
        "source": "Mishin et al. PRB 1999; CRC Handbook",
    },
    "Cu": {
        "name": "Copper", "structure": "fcc",
        "molar_mass_g_mol": 63.55,
        "lattice_constant_A": 3.615,
        "density_gcm3": 8.96,
        "C11_GPa": 168.0, "C12_GPa": 121.0, "C44_GPa": 75.0,
        "E_GPa": 128.0, "bulk_modulus_GPa": 140.0, "poisson_ratio": 0.34,
        "source": "Foiles et al. PRB 1986; CRC Handbook",
    },
    "W": {
        "name": "Tungsten", "structure": "bcc",
        "molar_mass_g_mol": 183.84,
        "lattice_constant_A": 3.165,
        "density_gcm3": 19.25,
        "C11_GPa": 523.0, "C12_GPa": 204.0, "C44_GPa": 161.0,
        "E_GPa": 411.0, "bulk_modulus_GPa": 310.0, "poisson_ratio": 0.28,
        "source": "Zhou et al. PRB 2004; CRC Handbook",
    },
}

# ---------------------------------------------------------------------------
# POLÍMEROS
# ---------------------------------------------------------------------------
POLYMER_REFERENCE = {
    "PE": {
        "name": "Polyethylene", "monomer_smiles": "CC",
        "density_gcm3": 0.95, "E_GPa": 0.8, "Tg_K": 153,
        "poisson_ratio": 0.42, "source": "CRC Handbook",
    },
    "PP": {
        "name": "Polypropylene", "monomer_smiles": "CC(C)",
        "density_gcm3": 0.91, "E_GPa": 1.5, "Tg_K": 263,
        "poisson_ratio": 0.42, "source": "CRC Handbook",
    },
    "PS": {
        "name": "Polystyrene", "monomer_smiles": "CC(c1ccccc1)",
        "density_gcm3": 1.05, "E_GPa": 3.2, "Tg_K": 373,
        "poisson_ratio": 0.34, "source": "CRC Handbook",
    },
    "PMMA": {
        "name": "Poly(methyl methacrylate)", "monomer_smiles": "CC(C)(C(=O)OC)",
        "density_gcm3": 1.18, "E_GPa": 3.3, "Tg_K": 388,
        "poisson_ratio": 0.40, "source": "CRC Handbook",
    },
    "PLA": {
        "name": "Polylactic acid", "monomer_smiles": "CC(O)C(=O)O",
        "density_gcm3": 1.24, "E_GPa": 3.5, "Tg_K": 333,
        "poisson_ratio": 0.40, "source": "CRC Handbook",
    },
}

# Mapa de nombres de propiedad -> clave en las tablas (acepta alias comunes).
_PROP_ALIASES = {
    "density": "density_gcm3", "density_gcm3": "density_gcm3", "rho": "density_gcm3",
    "e": "E_GPa", "e_gpa": "E_GPa", "young": "E_GPa", "young_modulus": "E_GPa",
    "modulus": "E_GPa", "youngs_modulus": "E_GPa",
    "tg": "Tg_K", "tg_k": "Tg_K", "glass_transition": "Tg_K",
    "lattice": "lattice_constant_A", "lattice_constant": "lattice_constant_A",
    "a": "lattice_constant_A",
    "bulk": "bulk_modulus_GPa", "bulk_modulus": "bulk_modulus_GPa", "k": "bulk_modulus_GPa",
    "poisson": "poisson_ratio", "nu": "poisson_ratio", "poisson_ratio": "poisson_ratio",
    "c11": "C11_GPa", "c12": "C12_GPa", "c44": "C44_GPa",
}


def get_reference(material: str, prop: str = None) -> dict:
    """Devuelve valor(es) de referencia curados para un material, con fuente.

    Args:
        material: símbolo de metal (Al, Ni, Cu, W) o código de polímero
                  (PE, PP, PS, PMMA, PLA). Case-insensitive; se compara EXACTO
                  (no substring), para no confundir 'Al' con 'materiAL'.
        prop: propiedad concreta (density, E, Tg, lattice_constant, ...).
              Si None, devuelve todas las propiedades del material.

    Returns:
        dict con: material, kind, source, y bien 'property'+'value'+'unit'
        (si se pidió prop) o 'properties' (dict completo). status='not_found'
        si el material no está en las tablas.
    """
    key = material.strip()
    # Búsqueda EXACTA (case-insensitive) en cada tabla
    table, kind = None, None
    for m, data in METAL_REFERENCE.items():
        if m.lower() == key.lower():
            table, kind = data, "metal"; break
    if table is None:
        for p, data in POLYMER_REFERENCE.items():
            if p.lower() == key.lower():
                table, kind = data, "polymer"; break

    if table is None:
        return {"material": material, "status": "not_found",
                "error": f"'{material}' not in curated reference tables. "
                         f"Available metals: {list(METAL_REFERENCE)}; "
                         f"polymers: {list(POLYMER_REFERENCE)}"}

    if prop is None:
        props = {k: v for k, v in table.items() if k not in ("name", "source",
                 "structure", "monomer_smiles")}
        return {"material": material, "name": table["name"], "kind": kind,
                "status": "ok", "source": table["source"], "properties": props}

    # propiedad concreta
    canon = _PROP_ALIASES.get(prop.strip().lower())
    if canon is None or canon not in table:
        return {"material": material, "status": "not_found",
                "error": f"Property '{prop}' not available for {material}. "
                         f"Have: {[k for k in table if k not in ('name','source','structure','monomer_smiles')]}",
                "source": table["source"]}
    unit = canon.split("_")[-1] if "_" in canon else ""
    return {"material": material, "name": table["name"], "kind": kind,
            "status": "ok", "property": canon, "value": table[canon],
            "unit": unit, "source": table["source"]}


# Compatibilidad hacia atrás: POLYMER_DB con el esquema que espera
# amber_functions (name, density_exp_gcm3, E_exp_GPa, Tg_exp_K, monomer_smiles).
POLYMER_DB = {
    code: {
        "name": d["name"],
        "monomer_smiles": d["monomer_smiles"],
        "density_exp_gcm3": d["density_gcm3"],
        "E_exp_GPa": d["E_GPa"],
        "Tg_exp_K": d["Tg_K"],
        "poisson_ratio": d.get("poisson_ratio", 0.40),
    }
    for code, d in POLYMER_REFERENCE.items()
}


if __name__ == "__main__":
    import json
    # smoke test
    print(json.dumps(get_reference("Al", "density"), indent=2))
    print(json.dumps(get_reference("PLA", "E"), indent=2))
    print(json.dumps(get_reference("PLA"), indent=2))
    print(json.dumps(get_reference("Xx", "density"), indent=2))
