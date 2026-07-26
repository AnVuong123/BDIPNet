import copy
import itertools
import linecache
import math
import sys
import time
import numpy as np
import logging
import os
import json

import torch
import inspect
import pandas as pd
import numpy as np
import itertools
from scipy.optimize import minimize
from ase.io import read, write
from ase import Atoms
from itertools import product
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar  # <-- thêm cái này
np.set_printoptions(precision=4, suppress=True)
#from sevenn.calculator import SevenNetCalculator
from ase.optimize import BFGS
from ase.optimize import FIRE
import copy
from collections import Counter

import signal
import itertools
import logging
import math
import time
import numpy as np
from scipy.optimize import minimize
from ase.calculators.mixing import SumCalculator
from mattersim.forcefield import MatterSimCalculator
import sys
from dftd3.ase import DFTD3
#from ase.calculators.dftd3 import DFTD3
import ast




# =========================
# 1. STRAIN + ENERGY
# =========================
import numpy as np
import itertools
from scipy.optimize import minimize
from ase.io import read, write
from ase.build import make_supercell
from ase import Atoms



import numpy as np
from scipy.optimize import minimize



def get_bottom_element(atoms):
    import numpy as np

    z = atoms.positions[:, 2]
    symbols = np.array(atoms.get_chemical_symbols())

    idx = np.argmax(z)   # atom thấp nhất
    return symbols[idx]
def get_top_element(atoms):
    import numpy as np

    z = atoms.positions[:, 2]
    symbols = np.array(atoms.get_chemical_symbols())

    idx = np.argmin(z)   # atom thấp nhất
    return symbols[idx]
def check_layer_symbols(bilayer_atoms, mono_top_atoms, mono_bot_atoms,z_cut,print_layer=False):
    import numpy as np
    from collections import Counter

    # ===== lấy z_cut =====
    print("z_cut =", z_cut)
    pos = bilayer_atoms.positions
    scaled = bilayer_atoms.get_scaled_positions()

    z = pos[:, 2]
    z_frac = scaled[:, 2]

    # ===== split layer =====
    bot_idx = np.where(z < z_cut)[0]
    top_idx = np.where(z >= z_cut)[0]

    bot_atoms = bilayer_atoms[bot_idx]
    top_atoms = bilayer_atoms[top_idx]
 
            
    top_set = set(top_atoms.get_chemical_symbols())
    bot_set = set(bot_atoms.get_chemical_symbols())

    mono_top_set = set(mono_top_atoms.get_chemical_symbols())
    mono_bot_set = set(mono_bot_atoms.get_chemical_symbols())
    # ===== check =====
    ok_top = top_set == mono_top_set
    ok_bot = bot_set == mono_bot_set

    return ok_top and ok_bot

def flip_z(atoms):
    scaled = atoms.get_scaled_positions()
    scaled[:, 2] = 1 - scaled[:, 2]
    atoms.set_scaled_positions(scaled)
    atoms.wrap()
    return atoms

def build_fast2(cif_top, cif_bot, t_top, t_bot,
                C_top, C_bot, distance,flip, vacuum=10):

    from ase.io import read
    from ase import Atoms
    import numpy as np

    atoms_top = read(cif_top)
    atoms_bot = read(cif_bot)
    if flip:
        atoms_top = flip_z(atoms_top)
        atoms_bot = flip_z(atoms_bot)
    if C_top is None:
        return None

    # ===== build supercell =====
    S_top = np.eye(3)
    S_bot = np.eye(3)

    S_top[:2, :2] = C_top
    S_bot[:2, :2] = C_bot



    super_top = make_supercell(atoms_top, S_top)
    super_bot = make_supercell(atoms_bot, S_bot)

    cell_top = super_top.cell.array.copy()
    cell_bot = super_bot.cell.array.copy()
    A_top = cell_top[:2, :2]
    A_bot = cell_bot[:2, :2]

    R = 0.5 * (A_top + A_bot)
    cell_top[:2, :2] = R
    cell_bot[:2, :2] = R

    super_top.set_cell(cell_top, scale_atoms=True)
    super_bot.set_cell(cell_bot, scale_atoms=True)

    # ===== cell bilayer =====
    cell3 = np.eye(3)
    cell3[:2, :2] = R
    cell3[2, 2] = vacuum + distance + t_top + t_bot
    # ===== align distance ban đầu =====
    super_bot.positions[:, 2] += -0
    z_top_min = super_top.positions[:, 2].min()
    z_bot_max = super_bot.positions[:, 2].max()
    #print("z_top_avg", z_top_avg)
    #print("z_bot_avg", z_bot_avg)
    #print("distance current", z_top_avg - z_bot_avg)
    z_d = z_top_min - z_bot_max
    #print(distance - z_avg,distance)
    super_top.positions[:, 2] += (distance - z_d) 

    positions = np.vstack([super_top.positions, super_bot.positions])
    z_top_min = super_top.positions[:, 2].min()
    z_bot_max = super_bot.positions[:, 2].max()

    # ===== compute z_cut =====
    z_cut = 0.5 * (z_top_min + z_bot_max)
    symbols = super_top.get_chemical_symbols() + super_bot.get_chemical_symbols()
    atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell3,
            pbc=True
        )
    return atoms,z_cut




import numpy as np
from ase.io import read, write

def compute_zcut(atoms):
    z = atoms.positions[:, 2]

    z_sorted = np.sort(z)

    dz = z_sorted[1:] - z_sorted[:-1]

    k = np.argmax(dz)

    zcut = (z_sorted[k] + z_sorted[k+1]) / 2

    return zcut 
def compute_z_cut_e(atoms, e_top, e_bot):
    import numpy as np

    symbols = np.array(atoms.get_chemical_symbols())
    z = atoms.positions[:, 2]

    z_top = z[symbols == e_top]
    z_bot = z[symbols == e_bot]

    if len(z_top) == 0 or len(z_bot) == 0:
        raise RuntimeError("❌ Missing atoms")


    dz_matrix = np.abs(z_top[:, None] - z_bot[None, :])

    i, j = np.unravel_index(np.argmin(dz_matrix), dz_matrix.shape)

    z_top_near = z_top[i]
    z_bot_near = z_bot[j]


    # ===== midpoint =====
    zcut = 0.5 * (z_top_near + z_bot_near)

    print("z_cut =", zcut)

    return zcut

def split_layers_by_z(atoms):
    z = atoms.positions[:, 2]
    zcut = compute_zcut(atoms)

    mask_top = z > zcut
    mask_bot = z <= zcut

    return mask_top, mask_bot


def get_surface_z(z):
    z_min = np.percentile(z, 5)
    z_max = np.percentile(z, 95)
    return z_min, z_max

def adjust_interlayer_distance_hetdb(cif_in, new_d, shift_frac=(0.0, 0.0)):
    atoms = read(cif_in)

    z = atoms.positions[:, 2]

    mask_top, mask_bot = split_layers_by_z(atoms)

    z_top = z[mask_top]
    z_bot = z[mask_bot]

    z_top_avg = z_top.mean()
    z_bot_avg = z_bot.mean()

    cell = atoms.cell.array.copy()
    c_old = cell[2, 2]

    d_old = z_top_avg - z_bot_avg


    c_new = c_old - d_old + new_d
    cell[2, 2] = c_new


    shift = new_d - d_old
    atoms.positions[mask_top, 2] += shift

    atoms.set_cell(cell, scale_atoms=False)
    atoms.wrap()


    z_cut = 0.5 * (z_top_avg + z_bot_avg)

    return atoms, z_cut

def adjust_interlayer_distance(cif_in, new_d, shift_frac=(0.0, 0.0)):
    atoms = read(cif_in)

    z = atoms.positions[:, 2]

    # ========================
    # split layer
    # ========================
    mask_top, mask_bot = split_layers_by_z(atoms)

    z_top = z[mask_top]
    z_bot = z[mask_bot]

   
    z_top_min = z_top.min()
    z_bot_max = z_bot.max()

   
    d_old = z_top_min - z_bot_max

  
    cell = atoms.cell.array.copy()
    c_old = cell[2, 2]

    c_new = c_old - d_old + new_d
    cell[2, 2] = c_new

    # ========================
    # shift top layer (z)
    # ========================
    shift_z = new_d - d_old
    atoms.positions[mask_top, 2] += shift_z

    # ========================
    # 🔥 XY SHIFT (NEW)
    # ========================
    ii, jj = shift_frac

    # lattice vectors (Cartesian)
    A1 = cell[0]
    A2 = cell[1]

    disp_xy = ii * A1 + jj * A2

    # chỉ apply cho top layer
    atoms.positions[mask_top, 0] += disp_xy[0]
    atoms.positions[mask_top, 1] += disp_xy[1]

    atoms.set_cell(cell, scale_atoms=False)
    atoms.wrap()


    z_cut = 0.5 * (z_top_min + z_bot_max)

    return atoms, z_cut


disp_A1 = [0.0, 1/8, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6]
disp_A2 = [0.0, 1/8, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6]


def xy_scan(d,db):
    best_energy = 1e9
    best_shift = (0, 0)

    for ii in disp_A1:
        for jj in disp_A2:
            energy = compute_energy(d,db, shift_frac=(ii, jj)) 
            print(f"xy: d={d:.3f}, shift=({ii:.3f},{jj:.3f}), E={energy:.6f}")

            if energy < best_energy:
                best_energy = energy
                best_shift = (ii, jj)

    return best_energy, best_shift

def z_scan(db,xy_s=True):
    d=0.5
    best_d=d
    best_energy=1000000
    best_shift = (0, 0)
    while d<=10:
        energy = compute_energy(d,db)
        print(d,energy)
        if energy <best_energy:
            best_energy=energy
            best_d=d
        d+=0.25

    step = 0.05
    current_d = best_d
    e0 = compute_energy(current_d,db)
    e_plus = compute_energy(current_d + step,db)
    e_minus = compute_energy(current_d - step,db)
    current_energy = e0
            
    print("refine:", current_d, e0, e_plus, e_minus)
    if e_plus < e0:
        direction = 1
        next_energy = e_plus
    elif e_minus < e0:
        direction = -1
        next_energy = e_minus
    else:
        direction=0
    print(e0,e_plus,e_minus,direction)
    if direction!=0:
        while True:
            next_d = current_d + direction*step 
            energy = compute_energy(next_d,db)
            print("walk:", next_d, energy,current_energy)
            if energy < current_energy:
                current_d = next_d
                current_energy = energy
            else:
                break
    print("🔥 FINAL:", current_d, current_energy)
    best_d=current_d
    best_energy=current_energy
    best_energy_xy, best_shift =0, (0, 0)
    if xy_s==True:
        best_energy_xy, best_shift = xy_scan(best_d,db)

        print("🔥 XY FINAL:", best_d, best_energy_xy, best_shift)
    return best_d, best_energy_xy, best_shift
def get_layers_with_element(atoms, indices, tol=1e-1):
    z = atoms.positions[:, 2]
    symbols = atoms.get_chemical_symbols()

    order = sorted(indices, key=lambda i: z[i])

    layers = []
    current = [order[0]]

    for i in order[1:]:
        z_mean = np.mean([z[j] for j in current])
        if abs(z[i] - z_mean) <= tol:
            current.append(i)
        else:
            layers.append(current)
            current = [i]
    layers.append(current)

    # build signature (z + element)
    sig = []
    for g in layers:
        z_mean = np.mean([z[i] for i in g])
        elem = tuple(sorted(Counter(symbols[i] for i in g).items()))
        sig.append((z_mean, elem))

    return sig

def extract_element_order(sig):
    """
    sig = [(z, (('Cl', 9),)), ...]
    → ['Cl', 'As', 'Te']
    """
    order = []
    for _, elem in sig:
      
        order.append(elem[0][0])
    return order

def is_flipped(atoms_bilayer, atoms_mono_bot):
    z = atoms_bilayer.positions[:, 2]
    z_cut = compute_zcut(atoms_bilayer)

    bot_idx = np.where(z <= z_cut)[0]

    sig_bi = get_layers_with_element(atoms_bilayer, bot_idx)
    sig_mono = get_layers_with_element(
        atoms_mono_bot, np.arange(len(atoms_mono_bot))
    )

    order_bi = extract_element_order(sig_bi)
    order_mono = extract_element_order(sig_mono)

    print("bilayer order:", order_bi)
    print("mono order   :", order_mono)

    if order_bi == order_mono:
        return False   
    elif order_bi == order_mono[::-1]:
        return True    
    else:
        print("⚠️ mismatch")
        return False



if __name__ == "__main__":

    mlip_calc = MatterSimCalculator(
        load_path="MatterSim-v1.0.0-5M.pth",
        device="cuda"   
    )
    # # ===== D3 =====
    d3 = DFTD3(
    method="pbe0",
    damping="d3zero",   # 🔥 zero damping

    realspace_cutoff={
        "disp2": 50.2,   # pair interaction (Å)
        "cn": 20.0       # coordination number (Å)
            }
        )

    # # ===== Combine: MLIP + D3 =====
    calc = SumCalculator([mlip_calc,d3])
    mae_list=[]   
    db="samba"
    #db="hetdb"
    #db="bidb"
    initial_bilayer_link="As2+NbTe2_abcd2ae4ee5a0c7d+3a172789_stack.cif" #db="samba"
    #initial_bilayer_link="1Ca4I8B13N13-1_stack.cif" #db="hetdb"
    #initial_bilayer_link="1C2O2V3-1-2-1_0_0_1--0.67_-0.33_stack.cif" #db=bidb
    atoms = read(initial_bilayer_link)
    
                   
    d=5
    best_d=d
    need_flip=False
    def compute_energy(d,db, shift_frac=(0.0, 0.0)):
            if db=="hetdb":
                atoms,z_cut =adjust_interlayer_distance_hetdb(initial_bilayer_link,d,shift_frac=shift_frac)
            else:
                atoms,z_cut =adjust_interlayer_distance(initial_bilayer_link,d,shift_frac=shift_frac)
            atoms.calc = calc
            return atoms.get_potential_energy()
    if db=="samba":
        xy_s=True
    else:
        xy_s=False
    best_d,best_e,best_shift=z_scan(db,xy_s=xy_s)
    if db=="hetdb":
        atoms,z_cut =adjust_interlayer_distance_hetdb(initial_bilayer_link,best_d,shift_frac=best_shift)
    else:
        atoms,z_cut =adjust_interlayer_distance(initial_bilayer_link,best_d,shift_frac=best_shift)     
    #Relaxation 
    atoms.calc = calc
    dyn = BFGS(atoms)
    if db=="hetdb":
        dyn.run(fmax=0.05)
    else:
        dyn.run(fmax=0.01)
    energy = atoms.get_potential_energy()
                    

    vasp_path = "bilayer.vasp"

    write(vasp_path,atoms,format="vasp",direct=False,vasp5=True)

    with open(vasp_path, "r") as f:
        lines = f.readlines()

        lines[0] = f"{z_cut:.6f}\n"

        with open(vasp_path, "w") as f:
            f.writelines(lines)
                
