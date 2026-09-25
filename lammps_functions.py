#!/usr/bin/env python
# coding: utf-8
"""
lammps_functions.py

Funciones LAMMPS puras extraídas de AtomAgents.py.
Sin dependencias de agentes LLM — solo Python, LAMMPS, ASE y numpy.
Tanto AtomAgents.py como AtomAgents_qwen.py importan de aquí.

Funciones incluidas:
  - create_crystal
  - create_potential_file
  - create_working_folder
  - lattice_constant_simulation
  - elastic_constant_simulation
  - surface_energy_simulation
  - stacking_fault_simulation
  - single_step_simulation
  - surface_energy_periodic
  - surface_energy_free
  - critical_stress_intensity_cleavage
  - gcd_of_vector
  - format_columns
  - process_atoms
  - random_alloy_solutes
  - find_central_atoms
"""
import os 
os.environ["MPLBACKEND"] = "Agg"

import re
import json
import math
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce
from typing import Annotated
from ase.build import bulk
from ase.io import lammpsdata
from ase.lattice.cubic import BodyCenteredCubic, FaceCenteredCubic
import atomman as am


# ---------------------------------------------------------------------------
# UTILIDADES AUXILIARES
# ---------------------------------------------------------------------------

def gcd_of_vector(vector):
    """Calcula el MCD de los componentes de un vector."""
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    return reduce(math.gcd, vector)


def format_columns(line, widths):
    """Formatea columnas con anchos fijos."""
    formatted_line = []
    for i, element in enumerate(line):
        formatted_line.append(f"{element:<{widths[i]}}")
    return ' '.join(formatted_line).rstrip() + '\n'


def process_atoms(working_folder, data_file, size_x, size_y):
    """Procesa átomos de un fichero de datos LAMMPS."""
    with open(f'./{working_folder}/{data_file}') as file:
        lines = file.readlines()

    init_index = None
    num_atoms_index = None

    for i, line in enumerate(lines):
        if re.search('Atoms', line):
            init_index = i
        if re.search('atoms', line):
            num_atoms_index = i
        if re.search('xlo xhi', line):
            num_x = i
        if re.search('ylo yhi', line):
            num_y = i

    if init_index is None or num_atoms_index is None:
        raise ValueError("No se encontraron los índices requeridos en el fichero.")

    xlo, xhi = float(lines[num_x].strip().split()[0]), float(lines[num_x].strip().split()[1])
    ylo, yhi = float(lines[num_y].strip().split()[0]), float(lines[num_y].strip().split()[1])

    xcent = (xlo + xhi)
    ycent = (ylo + yhi)
    unitx = (xhi - xlo) / size_x / 3
    unity = (yhi - ylo) / size_y / 2

    atom_cent_1 = [(xcent) / 2, (ycent - unity) / 2]
    atom_cent_2 = [(xcent) / 2 + unitx, (ycent - unity) / 2]

    num_atoms = int(lines[num_atoms_index].strip().split()[0])
    new_lines = []
    for i in range(init_index + 2, init_index + 2 + num_atoms):
        line = lines[i].rstrip().split()
        new_lines.append([int(line[0]), float(line[2]), float(line[3]), float(line[4])])

    return atom_cent_1, atom_cent_2, new_lines


def find_central_atoms(atoms, specific_point, threshold_distance=4):
    """Encuentra los átomos centrales de una dislocación."""
    df = pd.DataFrame(atoms, columns=['atom_id', 'atom_x', 'atom_y', 'atom_z'])
    df['distance'] = np.sqrt((df['atom_x'] - specific_point[0])**2 +
                             (df['atom_y'] - specific_point[1])**2)
    close_atoms = df[df['distance'] <= threshold_distance]
    assert len(close_atoms) >= 3, "Menos de 3 átomos cercanos encontrados"
    sorted_close_atoms = close_atoms.sort_values(by='distance')
    sorted_close_atom_ids = sorted_close_atoms['atom_id'].tolist()
    id_remove = []
    coords = sorted_close_atoms[['atom_x', 'atom_y']].values
    for i in range(len(sorted_close_atom_ids)):
        for j in range(i + 1, len(sorted_close_atom_ids)):
            if np.all(np.abs(coords[i] - coords[j]) < 0.01):
                id_remove.append(sorted_close_atom_ids[j])
    sorted_close_atom_ids = [id for id in sorted_close_atom_ids if id not in id_remove]
    id_central_atoms = sorted_close_atom_ids[:3]
    xxx = sorted_close_atoms.loc[sorted_close_atoms['atom_id'].isin(id_central_atoms), 'atom_x'].values
    yyy = sorted_close_atoms.loc[sorted_close_atoms['atom_id'].isin(id_central_atoms), 'atom_y'].values
    dislocation_cent = [np.mean(xxx), min(yyy) + 1/3 * (max(yyy) - min(yyy))]
    return dislocation_cent


def random_alloy_solutes(input_file, concentration_1):
    """Genera una aleación aleatoria modificando tipos de átomos."""
    with open(input_file, 'r') as file:
        lines = file.readlines()
    init_index = None
    num_atoms_index = None
    for i, line in enumerate(lines):
        if re.search('Atoms # atomic', line):
            init_index = i
        if re.search('atoms', line):
            num_atoms_index = i
    if init_index is None or num_atoms_index is None:
        raise ValueError("No se encontraron los índices requeridos.")
    num_atoms = int(lines[num_atoms_index].strip().split()[0])
    new_lines = []
    atom_indices = np.random.permutation(num_atoms)
    atom_types = np.zeros(num_atoms)
    atom_types[atom_indices[:int(num_atoms * concentration_1 / 100)]] = 1
    atom_types[atom_indices[int(num_atoms * concentration_1 / 100):num_atoms]] = 2
    new_lines.extend(lines[:init_index + 2])
    column_widths = [8, 4] + [15] * (len(lines[init_index + 2].split()) - 2)
    k = 0
    for i in range(init_index + 2, init_index + 2 + num_atoms):
        new_line = lines[i].rstrip().split()
        new_line[1] = str(int(atom_types[k]))
        new_lines.append(format_columns(new_line, column_widths))
        k += 1
    new_lines.extend(lines[init_index + 2 + num_atoms:])
    output_string = ''.join(new_lines)
    with open(input_file, 'w') as file:
        file.write(output_string)


# ---------------------------------------------------------------------------
# FUNCIONES PRINCIPALES LAMMPS
# ---------------------------------------------------------------------------

def create_working_folder(working_folder: str) -> str:
    """Crea una carpeta de trabajo para el proyecto."""
    try:
        os.mkdir(working_folder)
    except:
        pass
    return f"Working folder: {working_folder}"


def create_potential_file(working_folder: str,
                          pair_style_text: str,
                          pair_coeff_text: str) -> str:
    """Crea el fichero de potencial para simulaciones LAMMPS."""
    try:
        os.mkdir(working_folder)
    except:
        pass

    content = f'pair_style {pair_style_text} \npair_coeff {pair_coeff_text}'
    print(content)

    assert re.search('../potential_repository/', content), \
        "../potential_repository/ no encontrado"
    assert re.search(r'\* \*', content), "pair_coeff debe contener * *"

    if re.search('pair_coeff pair_coeff', content):
        raise TypeError("pair_coeff duplicado en el contenido")
    if re.search('pair_style pair_style', content):
        raise TypeError("pair_style duplicado en el contenido")

    filename = f'./{working_folder}/potential.inp'
    with open(filename, 'w') as file:
        file.write(content)
    return filename


def create_crystal(working_folder: str,
                   output_name: str,
                   lat_type: str,
                   lat_const: float,
                   conc_1: int,
                   conc_2: int,
                   mtp_pot: bool,
                   orient_x: list,
                   orient_y: list,
                   orient_z: list,
                   size_x: int,
                   size_y: int,
                   size_z: int,
                   element_1: str = 'X',
                   element_2: str = 'Y') -> str:
    """Crea una estructura cristalina para simulaciones LAMMPS."""
    try:
        os.mkdir(working_folder)
    except:
        pass

    if np.dot(orient_x, orient_y) == 0 and (np.dot(orient_x, orient_z) != 0 or np.dot(orient_y, orient_z) != 0):
        orient_3 = list(np.cross(orient_x, orient_y))
        gcd = gcd_of_vector(orient_3)
        if gcd != 0:
            orient_3 = [int(xxx / gcd) for xxx in orient_3]
        raise ValueError(f"Orientaciones no ortogonales. Usa orient_z={orient_3}")

    assert conc_1 + conc_2 == 100, 'La suma de concentraciones debe ser 100'

    file_output_name = f'./{working_folder}/{output_name}'
    try:
        os.remove(file_output_name)
    except FileNotFoundError:
        pass

    if conc_1 == 100:
        if lat_type == 'bcc':
            atoms = BodyCenteredCubic(directions=[orient_x, orient_y, orient_z],
                                      size=(size_x, size_y, size_z),
                                      symbol='X', latticeconstant=lat_const)
        elif lat_type == 'fcc':
            atoms = FaceCenteredCubic(directions=[orient_x, orient_y, orient_z],
                                      size=(size_x, size_y, size_z),
                                      symbol='X', latticeconstant=lat_const)
        lammpsdata.write_lammps_data(file_output_name, atoms)
        with open(file_output_name) as file:
            lines = file.readlines()
        new_lines = []
        for line in lines:
            if re.search(r'Atoms # atomic', line):
                new_lines.append('Masses\n\n')
                new_lines.append('1 1\n\n')
            new_lines.append(line)
        with open(file_output_name, 'w') as file:
            file.writelines(new_lines)

    if conc_2 == 100:
        assert mtp_pot == True, 'Para conc_2=100 se requiere mtp_pot=True'
        if lat_type == 'bcc':
            atoms = BodyCenteredCubic(directions=[orient_x, orient_y, orient_z],
                                      size=(size_x, size_y, size_z),
                                      symbol='X', latticeconstant=lat_const)
        elif lat_type == 'fcc':
            atoms = FaceCenteredCubic(directions=[orient_x, orient_y, orient_z],
                                      size=(size_x, size_y, size_z),
                                      symbol='X', latticeconstant=lat_const)
        atoms.symbols[0] = 'Nb'
        lammpsdata.write_lammps_data(file_output_name, atoms)
        with open(file_output_name) as file:
            lines = file.readlines()
        new_lines = []
        for line in lines:
            if re.search(r' 1   1', line):
                parts = re.split(r'(\s+)', line)
                parts[4] = '2'
                line = ''.join(parts)
            if re.search(r'Atoms # atomic', line):
                new_lines.append('Masses\n\n')
                from ase.data import atomic_masses, atomic_numbers
                m1 = atomic_masses[atomic_numbers.get(element_1, 1)]
                m2 = atomic_masses[atomic_numbers.get(element_2, 1)]
                new_lines.append(f'1 {m1}\n')
                new_lines.append(f'2 {m2}\n\n')
            new_lines.append(line)
        with open(file_output_name, 'w') as file:
            file.writelines(new_lines)

    if conc_1 != 100 and conc_1 != 0:
        if lat_type == 'bcc':
            atoms = BodyCenteredCubic(directions=[orient_x, orient_y, orient_z],
                                      size=(size_x, size_y, size_z),
                                      symbol='X', latticeconstant=lat_const)
        elif lat_type == 'fcc':
            atoms = FaceCenteredCubic(directions=[orient_x, orient_y, orient_z],
                                      size=(size_x, size_y, size_z),
                                      symbol='X', latticeconstant=lat_const)
        atom_indices = np.random.permutation(len(atoms))
        atoms.symbols[atom_indices[:int(len(atoms) * conc_1 / 100)]] = element_1
        atoms.symbols[atom_indices[int(len(atoms) * conc_1 / 100):len(atoms)]] = element_2
        lammpsdata.write_lammps_data(file_output_name, atoms)
        with open(file_output_name) as file:
            lines = file.readlines()
        new_lines = []
        for line in lines:
            if re.search(r'Atoms # atomic', line):
                new_lines.append('Masses\n\n')
                from ase.data import atomic_masses, atomic_numbers
                m1 = atomic_masses[atomic_numbers.get(element_1, 1)]
                m2 = atomic_masses[atomic_numbers.get(element_2, 1)]
                new_lines.append(f'1 {m1}\n')
                new_lines.append(f'2 {m2}\n\n')
            new_lines.append(line)
        with open(file_output_name, 'w') as file:
            file.writelines(new_lines)

    return output_name


def single_step_simulation(working_folder: str,
                           input_data_file_name: str,
                           size_x: int,
                           size_y: int,
                           size_z: int,
                           thermo_time: int = 100,
                           dump_time: int = 1000) -> dict:
    """Ejecuta una simulación de un solo paso para obtener parámetros de celda."""
    try:
        os.mkdir(working_folder)
    except:
        pass

    bash_script = f'''
clear
units metal
dimension 3
boundary p p p
atom_style atomic

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

read_data ${{data_file}}
include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo ${{thermo_time}}

variable p1 equal "step"
variable p2 equal "count(all)"
variable p3 equal "lx"
variable p4 equal "ly"
variable p5 equal "lz"
variable p6 equal "pe"
variable p7 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}} ${{p7}}" file "outputs.csv" title "#step numb_atoms a_x a_y a_z pe vol" screen no

reset_timestep 0
run 0
unfix extra
    '''

    lammps_script = f'{working_folder}/lmp_single_step_script.in'
    with open(lammps_script, 'w') as f:
        f.write(bash_script)

    text = f'''import sys
sys.path.insert(0, '/lustre/opt/foss/0.4/software/LAMMPS/29Aug2024_update2-foss-2023a-kokkos/lib/python3.11/site-packages')
from lammps import lammps
lmp = lammps(cmdargs=['-log', f'./{working_folder}/log.lammps.single.step'])
lmp.file(f'{working_folder}/lmp_single_step_script.in')
lmp.command('quit')
lmp.close()'''

    script_name = f'run_lammps_single_{working_folder}.py'
    with open(script_name, 'w') as f:
        f.write(text)

    command = f'mpirun -np 1 python {script_name}'
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    if output.returncode == 0:
        with open(f'./{working_folder}/outputs.csv') as file:
            lines = file.readlines()
        last_data = lines[-1].split(' ')
        num_atoms = int(last_data[1])
        Lx = float(last_data[2])
        Ly = float(last_data[3])
        Lz = float(last_data[4])
        pot_energy = float(last_data[5])
        vol = float(last_data[6])
        xlat = float(Lx / size_x)
        ylat = float(Ly / size_y)
        zlat = float(Lz / size_z)
        output_dict = {
            "number of atoms": f'{num_atoms}',
            "Lx (A)": f'{Lx}',
            "Ly (A)": f'{Ly}',
            "Lz (A)": f'{Lz}',
            "volume (A^3)": f'{vol}',
            "pot_energy (eV)": f'{pot_energy}',
            "xlat (A)": f'{xlat}',
            "ylat (A)": f'{ylat}',
            "zlat (A)": f'{zlat}',
        }
        return json.dumps(output_dict)
    else:
        stdout_text = output.stderr.strip().split('\n')
        return json.dumps(stdout_text, indent=4)


def lattice_constant_simulation(working_folder: str,
                                lat_type: str,
                                lat_const_guess: float,
                                conc_1: int,
                                conc_2: int,
                                mtp_pot: bool,
                                num_cpus: int = 4,
                                thermo_time: int = 100,
                                dump_time: int = 1000,
                                element_1: str = 'X',
                                element_2: str = 'Y') -> dict:
    """Calcula la constante de red por minimización de energía."""
    try:
        os.mkdir(working_folder)
    except:
        pass

    assert conc_1 + conc_2 == 100, 'Las concentraciones deben sumar 100.'

    if conc_1 != 0 and conc_1 != 100:
        total_samples = 3
    else:
        total_samples = 1

    nx = ny = nz = 10
    lat_all = []

    for num_samples in range(total_samples):
        if num_samples == 0:
            input_data_file_name = create_crystal(
                working_folder, 'lattice_constant_alloy.lmp',
                lat_type, lat_const_guess, conc_1, conc_2, mtp_pot,
                [1, 0, 0], [0, 1, 0], [0, 0, 1], nx, ny, nz,
                element_1=element_1, element_2=element_2)
        else:
            input_data_file_name = create_crystal(
                working_folder, 'lattice_constant_alloy.lmp',
                lat_type, np.mean(lat_all), conc_1, conc_2, mtp_pot,
                [1, 0, 0], [0, 1, 0], [0, 0, 1], nx, ny, nz,
                element_1=element_1, element_2=element_2)

        bash_script = f'''
clear
units metal
dimension 3
boundary p p p
atom_style atomic

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

read_data ${{data_file}}
include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol lx ly lz press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo ${{thermo_time}}

shell mkdir dump_lattice_constant

reset_timestep 0
dump first_dump all custom 1 dump.lattice.initial.{num_samples} id type x y z c_peratom
run 0
undump first_dump

variable dump_id equal 1
variable dump_out_name string "./dump_lattice_constant/dump.out"
dump ${{dump_id}} all custom ${{dump_time}} ${{dump_out_name}}.* id type x y z c_peratom

variable p1 equal "step"
variable p2 equal "count(all)"
variable p3 equal "lx"
variable p4 equal "ly"
variable p5 equal "lz"
variable p6 equal "pe"
variable p7 equal "vol"

fix 1 all box/relax aniso 0
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

min_style cg
minimize 0 1e-5 20000 20000

fix 1 all box/relax aniso 0
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

min_style cg
minimize 0 1e-5 20000 20000

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}} ${{p7}}" file "relaxed_outputs.csv" title "#step numb_atoms a_x a_y a_z pe vol" screen no

reset_timestep 0
dump last_dump all custom 1 dump.lattice.final.{num_samples} id type x y z c_peratom
run 0

unfix extra
undump last_dump
quit
        '''

        lammps_script = f'{working_folder}/lmp_lattice_constant_script.in'
        with open(lammps_script, 'w') as f:
            f.write(bash_script)

        text = f'''import sys
sys.path.insert(0, '/lustre/opt/foss/0.4/software/LAMMPS/29Aug2024_update2-foss-2023a-kokkos/lib/python3.11/site-packages')
from lammps import lammps
lmp = lammps(cmdargs=['-log', f'./{working_folder}/log.lammps.lattice.constant.alloy'])
lmp.file(f'{working_folder}/lmp_lattice_constant_script.in')
lmp.command('quit')
lmp.close()'''

        script_name = f'run_lammps_lattice_{working_folder}.py'
        with open(script_name, 'w') as f:
            f.write(text)

        command = f'mpirun -np {num_cpus} python {script_name}'
        output = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

        if output.returncode == 0:
            with open(f'./{working_folder}/relaxed_outputs.csv') as file:
                lines = file.readlines()
            last_data = lines[-1].split(' ')
            Lx = float(last_data[2])
            Ly = float(last_data[3])
            Lz = float(last_data[4])
            xlat = float(Lx / nx)
            ylat = float(Ly / ny)
            zlat = float(Lz / nz)
            lat_all.extend([xlat, ylat, zlat])
            output_dict = {'lattice constant': f'{np.mean(lat_all):.3f}'}
            return json.dumps(output_dict)
        else:
            stdout_text_1 = output.stderr.strip().split('\n')
            stdout_text_2 = output.stdout.strip().split('\n')[-2:]
            return json.dumps([stdout_text_1, stdout_text_2], indent=4)


def surface_energy_periodic(working_folder, input_data_file_name, thermo_time, dump_time):
    """Calcula la energía del sistema periódico para energía superficial."""
    bash_script = f'''
clear
units metal
dimension 3
boundary p p p
atom_style atomic

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

read_data ${{data_file}}
include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo ${{thermo_time}}

shell mkdir surface_energy

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "surface_energy_periodic.csv" title "#step a_x a_y a_z pe vol" screen no

min_style cg
minimize 0 1e-5 20000 20000

fix 1 all box/relax aniso 0
min_style cg
minimize 0 1e-5 2000 20000
unfix 1

reset_timestep 0
run 0
unfix extra

write_data data.surface.energy.periodic.lmp
    '''

    lammps_script = f'{working_folder}/lmp_surface_energy_periodic_script.in'
    with open(lammps_script, 'w') as f:
        f.write(bash_script)

    text = f'''import sys
sys.path.insert(0, '/lustre/opt/foss/0.4/software/LAMMPS/29Aug2024_update2-foss-2023a-kokkos/lib/python3.11/site-packages')
from lammps import lammps
lmp = lammps(cmdargs=['-in', '{working_folder}/lmp_surface_energy_periodic_script.in', '-log', f'./{working_folder}/log.lammps.surface_energy_periodic'])
lmp.file(f'{working_folder}/lmp_surface_energy_periodic_script.in')
lmp.command('quit')
lmp.close()'''

    script_name = f'run_lammps_surf_periodic_{working_folder}.py'
    with open(script_name, 'w') as f:
        f.write(text)

    command = f'mpirun -np 4 python {script_name}'
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    with open(f'./{working_folder}/surface_energy_periodic.csv') as file:
        lines = file.readlines()

    init_data = lines[-1].split(' ')
    pot_energy_initial = float(init_data[4])
    Lx = float(init_data[1])
    Ly = float(init_data[2])
    Lz = float(init_data[3])

    return pot_energy_initial, Lx, Ly, Lz, 'data.surface.energy.periodic.lmp'


def surface_energy_free(working_folder, input_data_file_name, surface_dir, thermo_time, dump_time):
    """Calcula la energía del sistema con superficie libre."""
    bash_script = f'''
clear
units metal
dimension 3
boundary p p p
atom_style atomic

shell cd ./{working_folder}

variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}
variable surface_dir string {surface_dir}
variable xxx string "x"
variable yyy string "y"
variable zzz string "z"

read_data ${{data_file}}
include ${{potential_file}}

compute peratom all pe/atom
compute pe all pe

thermo_style custom step temp pe vol press pxx pyy pzz fnorm
thermo_modify format float %5.5g
thermo ${{thermo_time}}

variable p1 equal "step"
variable p2 equal "lx"
variable p3 equal "ly"
variable p4 equal "lz"
variable p5 equal "pe"
variable p6 equal "vol"

fix extra all print 100 "${{p1}} ${{p2}} ${{p3}} ${{p4}} ${{p5}} ${{p6}}" file "surface_energy_free.csv" title "#step a_x a_y a_z pe vol" screen no

if "${{surface_dir}}==${{xxx}} " then "change_box all boundary s p p"
if "${{surface_dir}}==${{yyy}} " then "change_box all boundary p s p"
if "${{surface_dir}}==${{zzz}} " then "change_box all boundary p p s"

min_style cg
minimize 0 1e-5 20000 20000

reset_timestep 0
run 0
unfix extra
    '''

    lammps_script = f'{working_folder}/lmp_surface_energy_free_script.in'
    with open(lammps_script, 'w') as f:
        f.write(bash_script)

    text = f'''import sys
sys.path.insert(0, '/lustre/opt/foss/0.4/software/LAMMPS/29Aug2024_update2-foss-2023a-kokkos/lib/python3.11/site-packages')
from lammps import lammps
lmp = lammps(cmdargs=['-in', '{working_folder}/lmp_surface_energy_free_script.in', '-log', f'./{working_folder}/log.lammps.surface_energy_free'])
lmp.file(f'{working_folder}/lmp_surface_energy_free_script.in')
lmp.command('quit')
lmp.close()'''

    script_name = f'run_lammps_surf_free_{working_folder}.py'
    with open(script_name, 'w') as f:
        f.write(text)

    command = f'mpirun -np 1 python {script_name}'
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    with open(f'./{working_folder}/surface_energy_free.csv') as file:
        lines = file.readlines()

    final_data = lines[-1].split(' ')
    pot_energy_final = float(final_data[4])
    return pot_energy_final


def elastic_constant_simulation(working_folder: str,
                                lat_type: str,
                                a: float,
                                conc_1: int,
                                conc_2: int,
                                mtp_pot: bool,
                                thermo_time: int = 100,
                                dump_time: int = 1000,
                                element_1: str = 'X',
                                element_2: str = 'Y') -> dict:
    """Calcula las constantes elásticas C11, C12, C44."""
    try:
        os.mkdir(working_folder)
    except:
        pass

    assert conc_1 + conc_2 == 100, 'Las concentraciones deben sumar 100.'

    if conc_1 == 0 or conc_2 == 0:
        total_samples = 1
    else:
        total_samples = 3

    C_11_all, C_22_all, C_33_all = [], [], []
    C_12_all, C_13_all, C_23_all = [], [], []
    C_44_all, C_55_all, C_66_all = [], [], []

    for num_samples in range(total_samples):
        input_data_file_name = create_crystal(
            working_folder, 'elastic_constant.lmp',
            lat_type, a, conc_1, conc_2, mtp_pot,
            [1, 0, 0], [0, 1, 0], [0, 0, 1], 5, 5, 5,
            element_1=element_1, element_2=element_2)

        # Copiar displace.mod al directorio de trabajo
        import shutil
        shutil.copy('./0_codes/displace.mod', f'./{working_folder}/displace.mod')

        bash_script = f'''
variable data_file string "{input_data_file_name}"
variable potential_file string "potential.inp"
variable thermo_time equal {thermo_time}
variable dump_time equal {dump_time}

shell cp ./0_codes/displace.mod ./{working_folder}/displace.mod

shell cd ./{working_folder}

variable displace_file string "displace.mod"
variable up equal 1.0e-6
variable atomjiggle equal 1.0e-5

units metal
variable cfac equal 1.0e-4
variable cunits string GPa

variable etol equal 0.0
variable ftol equal 1.0e-5
variable maxiter equal 5000
variable maxeval equal 5000
variable dmax equal 1.0e-2

neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

boundary p p p

read_data ${{data_file}}
include ${{potential_file}}

change_box all triclinic

thermo 50
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no

min_style cg
min_modify dmax ${{dmax}} line quadratic

fix 3 all box/relax aniso 0.0
minimize ${{etol}} ${{ftol}} ${{maxiter}} ${{maxeval}}

variable tmp equal pxx
variable pxx0 equal ${{tmp}}
variable tmp equal pyy
variable pyy0 equal ${{tmp}}
variable tmp equal pzz
variable pzz0 equal ${{tmp}}
variable tmp equal pyz
variable pyz0 equal ${{tmp}}
variable tmp equal pxz
variable pxz0 equal ${{tmp}}
variable tmp equal pxy
variable pxy0 equal ${{tmp}}

variable tmp equal lx
variable lx0 equal ${{tmp}}
variable tmp equal ly
variable ly0 equal ${{tmp}}
variable tmp equal lz
variable lz0 equal ${{tmp}}

variable d1 equal -(v_pxx1-${{pxx0}})/(v_delta/v_len0)*${{cfac}}
variable d2 equal -(v_pyy1-${{pyy0}})/(v_delta/v_len0)*${{cfac}}
variable d3 equal -(v_pzz1-${{pzz0}})/(v_delta/v_len0)*${{cfac}}
variable d4 equal -(v_pyz1-${{pyz0}})/(v_delta/v_len0)*${{cfac}}
variable d5 equal -(v_pxz1-${{pxz0}})/(v_delta/v_len0)*${{cfac}}
variable d6 equal -(v_pxy1-${{pxy0}})/(v_delta/v_len0)*${{cfac}}

displace_atoms all random ${{atomjiggle}} ${{atomjiggle}} ${{atomjiggle}} 87287 units box

unfix 3
write_restart restart.equil

variable dir equal 1
include ${{displace_file}}
variable dir equal 2
include ${{displace_file}}
variable dir equal 3
include ${{displace_file}}
variable dir equal 4
include ${{displace_file}}
variable dir equal 5
include ${{displace_file}}
variable dir equal 6
include ${{displace_file}}

variable C11all equal ${{C11}}
variable C22all equal ${{C22}}
variable C33all equal ${{C33}}
variable C12all equal 0.5*(${{C12}}+${{C21}})
variable C13all equal 0.5*(${{C13}}+${{C31}})
variable C23all equal 0.5*(${{C23}}+${{C32}})
variable C44all equal ${{C44}}
variable C55all equal ${{C55}}
variable C66all equal ${{C66}}

variable C11cubic equal (${{C11all}}+${{C22all}}+${{C33all}})/3.0
variable C12cubic equal (${{C12all}}+${{C13all}}+${{C23all}})/3.0
variable C44cubic equal (${{C44all}}+${{C55all}}+${{C66all}})/3.0

fix extra all print 1 "${{C11all}} ${{C12all}} ${{C13all}} ${{C22all}} ${{C23all}} ${{C33all}} ${{C44all}} ${{C55all}} ${{C66all}}" file "elastic_constants.csv" title "#C11 C12 C13 C22 C23 C33 C44 C55 C66" screen no

run 0
        '''

        lammps_script = f'{working_folder}/lmp_elastic_constant_script.in'
        with open(lammps_script, 'w') as f:
            f.write(bash_script)

        text = f'''import sys
sys.path.insert(0, '/lustre/opt/foss/0.4/software/LAMMPS/29Aug2024_update2-foss-2023a-kokkos/lib/python3.11/site-packages')
from lammps import lammps
lmp = lammps(cmdargs=['-log', f'./{working_folder}/log.lammps.elastic_constants'])
lmp.file(f'{working_folder}/lmp_elastic_constant_script.in')
lmp.command('quit')
lmp.close()'''

        script_name = f'run_lammps_elastic_{working_folder}.py'
        with open(script_name, 'w') as f:
            f.write(text)

        command = f'mpirun -np 1 python {script_name}'
        output = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

        with open(f'./{working_folder}/elastic_constants.csv') as file:
            lines = file.readlines()

        # Buscar última línea con datos (no cabecera)
        data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
        if not data_lines:
            raise ValueError("No data lines in elastic_constants.csv")
        final_data = data_lines[-1].split()
        C11 = float(final_data[0])
        C12 = float(final_data[1])
        C13 = float(final_data[2])
        C22 = float(final_data[3])
        C23 = float(final_data[4])
        C33 = float(final_data[5])
        C44 = float(final_data[6])
        C55 = float(final_data[7])
        C66 = float(final_data[8])

        C_11_all.append(C11)
        C_22_all.append(C22)
        C_33_all.append(C33)
        C_12_all.append(C12)
        C_13_all.append(C13)
        C_23_all.append(C23)
        C_44_all.append(C44)
        C_55_all.append(C55)
        C_66_all.append(C66)

    output_dict = {
        "C11 (GPa)": f'{np.mean(C_11_all):.2f}',
        "C12 (GPa)": f'{np.mean(C_12_all):.2f}',
        "C13 (GPa)": f'{np.mean(C_13_all):.2f}',
        "C22 (GPa)": f'{np.mean(C_22_all):.2f}',
        "C23 (GPa)": f'{np.mean(C_23_all):.2f}',
        "C33 (GPa)": f'{np.mean(C_33_all):.2f}',
        "C44 (GPa)": f'{np.mean(C_44_all):.2f}',
        "C55 (GPa)": f'{np.mean(C_55_all):.2f}',
        "C66 (GPa)": f'{np.mean(C_66_all):.2f}',
    }
    return json.dumps(output_dict)


def surface_energy_simulation(working_folder: str,
                              lat_type: str,
                              lat_const: float,
                              conc_1: int,
                              conc_2: int,
                              mtp_pot: bool,
                              orient_x: list,
                              orient_y: list,
                              orient_z: list,
                              surface_orientation: list,
                              surface_dir: str,
                              L_periodic: float = 30,
                              L_free: float = 60,
                              thermo_time: int = 100,
                              dump_time: int = 1000,
                              element_1: str = 'X',
                              element_2: str = 'Y') -> dict:
    """Calcula la energía superficial para un plano cristalino dado."""
    try:
        os.mkdir(working_folder)
    except:
        pass

    assert conc_1 + conc_2 == 100, 'Las concentraciones deben sumar 100.'

    if conc_1 != 0 and conc_1 != 100:
        total_samples = 3
    else:
        total_samples = 1

    create_crystal(working_folder, 'unit_lattice.lmp',
                   lat_type, lat_const, conc_1, conc_2, mtp_pot,
                   orient_x, orient_y, orient_z, 10, 10, 10,
                   element_1=element_1, element_2=element_2)

    res_step = json.loads(single_step_simulation(working_folder, 'unit_lattice.lmp', 10, 10, 10))
    xlat = float(res_step['xlat (A)'])
    ylat = float(res_step['ylat (A)'])
    zlat = float(res_step['zlat (A)'])

    if surface_dir == 'x':
        nx = int(np.ceil(L_free / xlat))
        ny = int(np.ceil(L_periodic / ylat))
        nz = int(np.ceil(L_periodic / zlat))
    elif surface_dir == 'y':
        nx = int(np.ceil(L_periodic / xlat))
        ny = int(np.ceil(L_free / ylat))
        nz = int(np.ceil(L_periodic / zlat))
    elif surface_dir == 'z':
        nx = int(np.ceil(L_periodic / xlat))
        ny = int(np.ceil(L_periodic / ylat))
        nz = int(np.ceil(L_free / zlat))

    surface_energy_all = []
    surface_energy_mj_all = []

    for num_samples in range(total_samples):
        input_data_file_name = create_crystal(
            working_folder, 'surface_energy.lmp',
            lat_type, lat_const, conc_1, conc_2, mtp_pot,
            orient_x, orient_y, orient_z, nx, ny, nz,
            element_1=element_1, element_2=element_2)

        pot_energy_init, Lx, Ly, Lz, relaxed_structure = surface_energy_periodic(
            working_folder, input_data_file_name, thermo_time, dump_time)

        pot_energy_final = surface_energy_free(
            working_folder, relaxed_structure, surface_dir, thermo_time, dump_time)

        if surface_dir == 'x':
            surface_energy = (pot_energy_final - pot_energy_init) / (2 * Ly * Lz)
        elif surface_dir == 'y':
            surface_energy = (pot_energy_final - pot_energy_init) / (2 * Lx * Lz)
        elif surface_dir == 'z':
            surface_energy = (pot_energy_final - pot_energy_init) / (2 * Lx * Ly)

        surface_energy_mj = surface_energy * 16.0218
        surface_energy_all.append(surface_energy)
        surface_energy_mj_all.append(surface_energy_mj)

    output_dict = {
        'surface energy (eV/A^2)': f'{np.mean(surface_energy_all):.3f}',
        'surface energy (J/m^2)': f'{np.mean(surface_energy_mj_all):.3f}'
    }
    return json.dumps(output_dict)


def critical_stress_intensity_cleavage(orient_x: list,
                                       orient_y: list,
                                       orient_z: list,
                                       c11: float,
                                       c12: float,
                                       c44: float,
                                       gamma_s: float) -> str:
    """Calcula la tenacidad de fractura crítica KIc usando LEFM."""
    e1 = [1, 0, 0]
    e2 = [0., 1., 0.]
    e3 = [0, 0, 1]

    m1 = np.array(orient_x, dtype=float)
    m2 = np.array(orient_y, dtype=float)
    m3 = np.array(orient_z, dtype=float)

    m1 /= np.linalg.norm(m1)
    m2 /= np.linalg.norm(m2)
    m3 /= np.linalg.norm(m3)

    Q = np.array([[np.dot(m1, e1), np.dot(m1, e2), np.dot(m1, e3)],
                  [np.dot(m2, e1), np.dot(m2, e2), np.dot(m2, e3)],
                  [np.dot(m3, e1), np.dot(m3, e2), np.dot(m3, e3)]])

    K1 = np.array([[Q[0,0]**2, Q[0,1]**2, Q[0,2]**2],
                   [Q[1,0]**2, Q[1,1]**2, Q[1,2]**2],
                   [Q[2,0]**2, Q[2,1]**2, Q[2,2]**2]])
    K2 = np.array([[Q[0,1]*Q[0,2], Q[0,2]*Q[0,0], Q[0,0]*Q[0,1]],
                   [Q[1,1]*Q[1,2], Q[1,2]*Q[1,0], Q[1,0]*Q[1,1]],
                   [Q[2,1]*Q[2,2], Q[2,2]*Q[2,0], Q[2,0]*Q[2,1]]])
    K3 = np.array([[Q[1,0]*Q[2,0], Q[1,1]*Q[2,1], Q[1,2]*Q[2,2]],
                   [Q[2,0]*Q[0,0], Q[2,1]*Q[0,1], Q[2,2]*Q[0,2]],
                   [Q[0,0]*Q[1,0], Q[0,1]*Q[1,1], Q[0,2]*Q[1,2]]])
    K4 = np.array([[Q[1,1]*Q[2,2]+Q[1,2]*Q[2,1], Q[1,2]*Q[2,0]+Q[1,0]*Q[2,2], Q[1,0]*Q[2,1]+Q[1,1]*Q[2,0]],
                   [Q[2,1]*Q[0,2]+Q[2,2]*Q[0,1], Q[2,2]*Q[0,0]+Q[2,0]*Q[0,2], Q[2,0]*Q[0,1]+Q[2,1]*Q[0,0]],
                   [Q[0,1]*Q[1,2]+Q[0,2]*Q[1,1], Q[0,2]*Q[1,0]+Q[0,0]*Q[1,2], Q[0,0]*Q[1,1]+Q[0,1]*Q[1,0]]])

    KK1 = np.concatenate((K1, 2*K2), axis=1)
    KK2 = np.concatenate((K3, K4), axis=1)
    KK = np.concatenate((KK1, KK2), axis=0)

    C_base = np.array([[c11, c12, c12, 0, 0, 0],
                       [c12, c11, c12, 0, 0, 0],
                       [c12, c12, c11, 0, 0, 0],
                       [0, 0, 0, c44, 0, 0],
                       [0, 0, 0, 0, c44, 0],
                       [0, 0, 0, 0, 0, c44]])

    C = np.dot(np.dot(KK, C_base), np.transpose(KK))
    A1 = np.linalg.inv(C)

    QQ = np.array([[C[0,0], C[0,5], C[0,4]],
                   [C[0,5], C[5,5], C[4,5]],
                   [C[0,4], C[4,5], C[4,4]]])
    R = np.array([[C[0,5], C[0,1], C[0,3]],
                  [C[5,5], C[1,5], C[3,5]],
                  [C[4,5], C[1,4], C[3,4]]])
    T = np.array([[C[5,5], C[1,5], C[3,5]],
                  [C[1,5], C[1,1], C[1,3]],
                  [C[3,5], C[1,3], C[3,3]]])

    N1 = -1 * np.dot(np.linalg.inv(T), np.transpose(R))
    N2 = np.linalg.inv(T)
    N3 = np.dot(np.dot(R, np.linalg.inv(T)), np.transpose(R)) - QQ

    NN1 = np.concatenate((N1, N2), axis=1)
    NN2 = np.concatenate((N3, np.transpose(N1)), axis=1)
    N = np.concatenate((NN1, NN2), axis=0)

    [v, u] = np.linalg.eig(N)

    a1 = [[u[0,0]], [u[1,0]], [u[2,0]]]
    pp1 = v[0]
    b1 = np.dot(np.transpose(R) + np.dot(pp1, T), a1)

    a2 = [[u[0,2]], [u[1,2]], [u[2,2]]]
    pp2 = v[2]
    b2 = np.dot(np.transpose(R) + np.dot(pp2, T), a2)

    a3 = [[u[0,4]], [u[1,4]], [u[2,4]]]
    pp3 = v[4]
    b3 = np.dot(np.transpose(R) + np.dot(pp3, T), a3)

    AA = np.concatenate((a1, a2, a3), axis=1)
    BB = np.concatenate((b1, b2, b3), axis=1)

    L = 0.5 * np.real(1j * np.dot(AA, np.linalg.inv(BB)))
    lambda_coeff = np.linalg.inv(L)[1, 1]
    K_GG = np.sqrt(2 * gamma_s * lambda_coeff * 10**9) * 10**(-6)

    output_dict = {'Critical fracture toughness (MPa*m^1/2)': f'{K_GG:.3f}'}
    return json.dumps(output_dict)


def compute_young_modulus(C11: float, C12: float, C44: float) -> dict:
    """Calcula el módulo de Young E y módulo de cizalla G para un cristal cúbico
    usando la aproximación de Voigt-Reuss-Hill (VRH) para policristales.
    
    Args:
        C11, C12, C44: Constantes elásticas en GPa
    Returns:
        dict con E (GPa), G (GPa), nu (ratio de Poisson), B (módulo de volumen GPa)
    """
    import numpy as np
    
    # Módulo de volumen B
    B = (C11 + 2 * C12) / 3.0
    
    # Módulo de cizalla G — aproximación VRH
    G_voigt = (C11 - C12 + 3 * C44) / 5.0
    G_reuss = 5 * C44 * (C11 - C12) / (4 * C44 + 3 * (C11 - C12))
    G = (G_voigt + G_reuss) / 2.0
    
    # Módulo de Young E y ratio de Poisson nu
    E = 9 * B * G / (3 * B + G)
    nu = (3 * B - 2 * G) / (2 * (3 * B + G))
    
    return {
        "B (GPa)": round(B, 2),
        "G (GPa)": round(G, 2),
        "E (GPa)": round(E, 2),
        "nu": round(nu, 4),
    }


def compute_metal_density(lattice_constant_A: float, element: str,
                          structure: str = None) -> dict:
    """Densidad de un metal DERIVADA de su constante de red calculada.

    rho = n * M / (N_A * a^3)
      n = átomos por celda unidad (fcc=4, bcc=2, hcp=2 aprox, diamond=8)
      M = masa molar (g/mol), a = constante de red calculada (cm)

    Es un cálculo DERIVADO de un resultado de simulación LAMMPS, no un
    valor de tabla. source="LAMMPS_derived". La masa molar y (si no se
    pasa) la estructura se toman de reference_data (curado).

    Args:
        lattice_constant_A: constante de red en Angstrom (de LAMMPS).
        element: símbolo del metal (Al, Ni, Cu, W...).
        structure: fcc/bcc/hcp/diamond. Si None, se toma de reference_data.
    Returns:
        dict con density_gcm3, source, y los parámetros usados (trazable).
    """
    N_A = 6.02214076e23
    ATOMS_PER_CELL = {"fcc": 4, "bcc": 2, "hcp": 2, "diamond": 8, "sc": 1}

    # Masa molar y estructura desde reference_data (fuente única)
    molar_mass = None
    try:
        from reference_data import METAL_REFERENCE
        ref = None
        for m, d in METAL_REFERENCE.items():
            if m.lower() == element.lower():
                ref = d; break
        if ref:
            molar_mass = ref.get("molar_mass_g_mol")
            if structure is None:
                structure = ref.get("structure")
    except Exception:
        pass

    if molar_mass is None:
        return {"status": "failed",
                "error": f"No molar mass for '{element}' in reference_data; "
                         f"cannot derive density"}
    if structure is None:
        return {"status": "failed",
                "error": f"Structure unknown for '{element}'; pass structure=fcc/bcc/..."}

    n = ATOMS_PER_CELL.get(structure.lower())
    if n is None:
        return {"status": "failed",
                "error": f"Unknown structure '{structure}'. "
                         f"Known: {list(ATOMS_PER_CELL)}"}

    a_cm = lattice_constant_A * 1e-8  # Angstrom -> cm
    volume_cm3 = a_cm ** 3
    density = (n * molar_mass) / (N_A * volume_cm3)

    return {
        "status": "completed",
        "element": element,
        "density_gcm3": round(density, 4),
        "source": "LAMMPS_derived",
        "method": "rho = n*M/(N_A*a^3) from computed lattice constant",
        "lattice_constant_A": lattice_constant_A,
        "structure": structure,
        "atoms_per_cell": n,
        "molar_mass_g_mol": molar_mass,
    }
