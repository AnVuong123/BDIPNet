import itertools
import random
import sys
import time
from pathlib import Path
from typing import Optional

import os
import torch
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
import torch
import torch.nn.functional as F
from jarvis.core.graphs import nearest_neighbor_edges, build_undirected_edgedata
from jarvis.db.figshare import data as jdata
from jarvis.core.specie import chem_data, get_node_attributes

from collections import defaultdict
import numpy as np
from jarvis.core.graphs import canonize_edge
# from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson

from pandarallel import pandarallel
import periodictable


pandarallel.initialize(progress_bar=True)

tqdm.pandas()

torch.set_printoptions(precision=10)


def find_index_array(A, B):
    _, n = B.shape
    index_array = torch.zeros(n, dtype=torch.long)

    for i in range(n):
        idx = torch.where((A == B[:, i].unsqueeze(1)).all(dim=0))[0]
        index_array[i] = idx

    return index_array

class StructureDataset(InMemoryDataset):

    def __init__(self, df, data_path, processdir, target, name, atom_features="atomic_number",
                 id_tag="jid", root='./', transform=None, pre_transform=None, pre_filter=None,
                 mean=None, std=None, normalize=False):
        
        self.df = df
        self.data_path = data_path
        self.processdir = processdir
        self.target = target
        self.name = name
        self.atom_features = atom_features
        self.id_tag = id_tag
        self.ids = self.df[self.id_tag]
        self.labels = torch.tensor(self.df[self.target]).type(
              torch.get_default_dtype()
         )
  
        #self.mat_name = self.df["mat_name"]
        #self.m_cols = [f"m_{j}" for j in range(256)]
   
        
        self.target=torch.tensor(self.df[self.target]).type(
              torch.get_default_dtype()
         )
        if mean is not None:
            self.mean = mean
        elif normalize:
            self.mean = torch.mean(self.labels)
        else:
            self.mean = 0.0
        if std is not None:
            self.std = std
        elif normalize:
            self.std = torch.std(self.labels)
        else:
            self.std = 1.0

        self.group_id = {
            "H": 0,
            "He": 1,
            "Li": 2,
            "Be": 3,
            "B": 4,
            "C": 0,
            "N": 0,
            "O": 0,
            "F": 5,
            "Ne": 1,
            "Na": 2,
            "Mg": 3,
            "Al": 6,
            "Si": 4,
            "P": 0,
            "S": 0,
            "Cl": 5,
            "Ar": 1,
            "K": 2,
            "Ca": 3,
            "Sc": 7,
            "Ti": 7,
            "V": 7,
            "Cr": 7,
            "Mn": 7,
            "Fe": 7,
            "Co": 7,
            "Ni": 7,
            "Cu": 7,
            "Zn": 7,
            "Ga": 6,
            "Ge": 4,
            "As": 4,
            "Se": 0,
            "Br": 5,
            "Kr": 1,
            "Rb": 2,
            "Sr": 3,
            "Y": 7,
            "Zr": 7,
            "Nb": 7,
            "Mo": 7,
            "Tc": 7,
            "Ru": 7,
            "Rh": 7,
            "Pd": 7,
            "Ag": 7,
            "Cd": 7,
            "In": 6,
            "Sn": 6,
            "Sb": 4,
            "Te": 4,
            "I": 5,
            "Xe": 1,
            "Cs": 2,
            "Ba": 3,
            "La": 8,
            "Ce": 8,
            "Pr": 8,
            "Nd": 8,
            "Pm": 8,
            "Sm": 8,
            "Eu": 8,
            "Gd": 8,
            "Tb": 8,
            "Dy": 8,
            "Ho": 8,
            "Er": 8,
            "Tm": 8,
            "Yb": 8,
            "Lu": 8,
            "Hf": 7,
            "Ta": 7,
            "W": 7,
            "Re": 7,
            "Os": 7,
            "Ir": 7,
            "Pt": 7,
            "Au": 7,
            "Hg": 7,
            "Tl": 6,
            "Pb": 6,
            "Bi": 6,
            "Po": 4,
            "At": 5,
            "Rn": 1,
            "Fr": 2,
            "Ra": 3,
            "Ac": 9,
            "Th": 9,
            "Pa": 9,
            "U": 9,
            "Np": 9,
            "Pu": 9,
            "Am": 9,
            "Cm": 9,
            "Bk": 9,
            "Cf": 9,
            "Es": 9,
            "Fm": 9,
            "Md": 9,
            "No": 9,
            "Lr": 9,
            "Rf": 7,
            "Db": 7,
            "Sg": 7,
            "Bh": 7,
            "Hs": 7
        }

        super(StructureDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.process() 
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.path.join(self.root, self.data_path)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processdir)

    @property
    def processed_file_names(self):
        return self.name + '.pt'

    def process(self):
        
        mat_data = torch.load(self.raw_file_names)

        data_list = []
        features = self._get_attribute_lookup(self.atom_features)

        assert len(mat_data) == len(self.labels), \
            f"Mismatch: mat_data={len(mat_data)}, labels={len(self.labels)}"
        for i in tqdm(range(len(mat_data))):
            if mat_data[i] is None:
                print(f"Sample {i} is None")
                continue
            z = mat_data[i].x
            mat_data[i].atom_numbers = z

            group_feats = []
            for atom in z:
                group_feats.append(self.group_id[periodictable.elements[int(atom)].symbol])
            group_feats = torch.tensor(np.array(group_feats)).type(torch.LongTensor)
            identity_matrix = torch.eye(10)
            g_feats = identity_matrix[group_feats]
            if len(list(g_feats.size())) == 1:
                g_feats = g_feats.unsqueeze(0)

            f = torch.tensor(features[mat_data[i].atom_numbers.long().squeeze(1)]).type(torch.FloatTensor)
            if len(mat_data[i].atom_numbers) == 1:
                f = f.unsqueeze(0)

            mat_data[i].x = f
            mat_data[i].g_feats = g_feats
            mat_data[i].y  = (self.labels[i] - self.mean) / self.std
            #print(self.mean,self.std)
           
            
            #mat_data[i].mat_name = self.mat_name[i]
    
            mat_data[i].label =  self.labels[i]  #self.labels[i]
            data_list.append(mat_data[i])
        

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        mat_data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((mat_data, slices), self.processed_paths[0])

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        max_z = max(v["Z"] for v in chem_data.values())

        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features


class StructureDataset2(InMemoryDataset):

    def __init__(
        self,
        df,
        data_path,
        processdir,
        target,
        name,
        atom_features="atomic_number",
        id_tag="jid",
        root="./",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        mean=None,
        std=None,
        normalize=False,
    ):

        self.df = df
        self.data_path = data_path
        self.processdir = processdir
        self.target = target
        self.name = name
        self.atom_features = atom_features
        self.id_tag = id_tag

        self.ids = self.df[self.id_tag]

        self.labels = torch.tensor(
            self.df[self.target]
        ).type(torch.get_default_dtype())

        self.mono_target = torch.tensor(
            self.df["mono_target"]
        ).type(torch.get_default_dtype())

        self.mono_vectors = self.df["mono_vector"]

        if mean is not None:
            self.mean = mean
        elif normalize:
            self.mean = torch.mean(self.labels)
        else:
            self.mean = 0.0

        if std is not None:
            self.std = std
        elif normalize:
            self.std = torch.std(self.labels)
        else:
            self.std = 1.0

        self.group_id = {
            "H": 0, "He": 1, "Li": 2, "Be": 3, "B": 4,
            "C": 0, "N": 0, "O": 0, "F": 5, "Ne": 1,
            "Na": 2, "Mg": 3, "Al": 6, "Si": 4, "P": 0,
            "S": 0, "Cl": 5, "Ar": 1, "K": 2, "Ca": 3,
            "Sc": 7, "Ti": 7, "V": 7, "Cr": 7, "Mn": 7,
            "Fe": 7, "Co": 7, "Ni": 7, "Cu": 7, "Zn": 7,
            "Ga": 6, "Ge": 4, "As": 4, "Se": 0, "Br": 5,
            "Kr": 1, "Rb": 2, "Sr": 3, "Y": 7, "Zr": 7,
            "Nb": 7, "Mo": 7, "Tc": 7, "Ru": 7, "Rh": 7,
            "Pd": 7, "Ag": 7, "Cd": 7, "In": 6, "Sn": 6,
            "Sb": 4, "Te": 4, "I": 5, "Xe": 1, "Cs": 2,
            "Ba": 3, "La": 8, "Ce": 8, "Pr": 8, "Nd": 8,
            "Pm": 8, "Sm": 8, "Eu": 8, "Gd": 8, "Tb": 8,
            "Dy": 8, "Ho": 8, "Er": 8, "Tm": 8, "Yb": 8,
            "Lu": 8, "Hf": 7, "Ta": 7, "W": 7, "Re": 7,
            "Os": 7, "Ir": 7, "Pt": 7, "Au": 7, "Hg": 7,
            "Tl": 6, "Pb": 6, "Bi": 6, "Po": 4, "At": 5,
            "Rn": 1, "Fr": 2, "Ra": 3, "Ac": 9, "Th": 9,
            "Pa": 9, "U": 9, "Np": 9, "Pu": 9, "Am": 9,
            "Cm": 9, "Bk": 9, "Cf": 9, "Es": 9, "Fm": 9,
            "Md": 9, "No": 9, "Lr": 9, "Rf": 7, "Db": 7,
            "Sg": 7, "Bh": 7, "Hs": 7,
        }

        super(StructureDataset, self).__init__(
            root,
            transform,
            pre_transform,
            pre_filter
        )

        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.path.join(self.root, self.data_path)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processdir)

    @property
    def processed_file_names(self):
        return self.name + ".pt"

    def process(self):

        mat_data = torch.load(self.raw_file_names)

        data_list = []
        features = self._get_attribute_lookup(self.atom_features)

        assert len(mat_data) == len(self.labels), \
            f"Mismatch: mat_data={len(mat_data)}, labels={len(self.labels)}"

        identity_matrix = torch.eye(10)

        for i in tqdm(range(len(mat_data))):
            if mat_data[i] is None:
                print(f"Sample {i} is None")
                continue

            data = mat_data[i]

            # =====================================================
            # TOP layer
            # =====================================================
            z_top = data.x_top
            data.atom_numbers_top = z_top

            group_feats_top = []
            for atom in z_top:
                symbol = periodictable.elements[int(atom)].symbol
                group_feats_top.append(self.group_id[symbol])

            group_feats_top = torch.tensor(
                np.array(group_feats_top)
            ).type(torch.LongTensor)

            g_feats_top = identity_matrix[group_feats_top]

            if len(list(g_feats_top.size())) == 1:
                g_feats_top = g_feats_top.unsqueeze(0)

            f_top = torch.tensor(
                features[data.atom_numbers_top.long().squeeze(1)]
            ).type(torch.FloatTensor)

            if len(data.atom_numbers_top) == 1:
                f_top = f_top.unsqueeze(0)

            data.x_top = f_top
            data.g_feats_top = g_feats_top

            # =====================================================
            # BOT layer
            # =====================================================
            z_bot = data.x_bot
            data.atom_numbers_bot = z_bot

            group_feats_bot = []
            for atom in z_bot:
                symbol = periodictable.elements[int(atom)].symbol
                group_feats_bot.append(self.group_id[symbol])

            group_feats_bot = torch.tensor(
                np.array(group_feats_bot)
            ).type(torch.LongTensor)

            g_feats_bot = identity_matrix[group_feats_bot]

            if len(list(g_feats_bot.size())) == 1:
                g_feats_bot = g_feats_bot.unsqueeze(0)

            f_bot = torch.tensor(
                features[data.atom_numbers_bot.long().squeeze(1)]
            ).type(torch.FloatTensor)

            if len(data.atom_numbers_bot) == 1:
                f_bot = f_bot.unsqueeze(0)

            data.x_bot = f_bot
            data.g_feats_bot = g_feats_bot

            # =====================================================
            # labels / mono vector
            # =====================================================
            data.y = (self.labels[i] - self.mean) / self.std
            data.mono_target = self.mono_target[i]

            mono_vector = self.mono_vectors[i]

            if isinstance(mono_vector, torch.Tensor):
                data.mono_vector = mono_vector.float().view(1, -1)
            else:
                data.mono_vector = torch.tensor(
                    mono_vector,
                    dtype=torch.float32
                ).view(1, -1)

            data.label = self.labels[i]

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [
                data for data in data_list
                if self.pre_filter(data)
            ]

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(data)
                for data in data_list
            ]

        mat_data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((mat_data, slices), self.processed_paths[0])

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        max_z = max(v["Z"] for v in chem_data.values())

        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features
def nearest_neighbor_edges_vdw(
    atoms=None,
    cutoff=4,
    max_neighbors=12,
    z_cut=None,
    id=None,
    use_canonize=False,
    max_cutoff=100,
    tol=1e-6,
):

    coords = atoms.cart_coords
    z_layer = coords[:, 2] > z_cut
    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    # ---------- helper ----------
    def keep_full_shell(neighborlist, k):
        if len(neighborlist) <= k:
            return neighborlist

        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        kth_dist = neighborlist[k - 1][2]

        return [
            nbr for nbr in neighborlist
            if nbr[2] <= kth_dist + tol
        ]

    # ---------- check if need expand (shell-aware interlayer) ----------
    for site_idx, neighborlist in enumerate(all_neighbors):

        inter = [
            nbr for nbr in neighborlist
            if z_layer[site_idx] != z_layer[nbr[1]]
        ]

        inter_sorted = sorted(inter, key=lambda x: x[2])
        inter_shell = keep_full_shell(inter_sorted, max_neighbors)

        if len(inter_shell) < max_neighbors:
            new_cutoff = 2.0 * cutoff
            if new_cutoff > max_cutoff:
                print("Reached max_cutoff. Stop expanding.")
                return {}, cutoff

            return nearest_neighbor_edges_vdw(
                atoms=atoms,
                cutoff=new_cutoff,
                max_neighbors=max_neighbors,
                z_cut=z_cut,
                id=id,
                use_canonize=use_canonize,
                max_cutoff=max_cutoff,
                tol=tol,
            )

    # ---------- build edges ----------
    edges = defaultdict(set)

    for site_idx, neighborlist in enumerate(all_neighbors):

        inter = [
            nbr for nbr in neighborlist
            if z_layer[site_idx] != z_layer[nbr[1]]
        ]

        inter_sorted = sorted(inter, key=lambda x: x[2])
        inter_shell = keep_full_shell(inter_sorted, max_neighbors)

        for nbr in inter_shell:

            dst = nbr[1]
            image = tuple(nbr[3])

            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), image
            )

            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(image)

    return edges, cutoff
def nearest_neighbor_edges_vdw_full(
    atoms=None,
    cutoff=4,
    max_neighbors=12,
    z_cut=None,
    id=None,
    use_canonize=False,
    max_cutoff=100,
    tol=1e-6,
):
    coords = atoms.cart_coords
    z_layer = coords[:, 2] > z_cut

    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    # Remove neighbors coming from periodic images along the z direction.
    # Keep only neighbors whose image shift has zero z-component.
    all_neighbors = [
        [
            nbr for nbr in neighborlist
            if tuple(nbr[3])[2] == 0
        ]
        for neighborlist in all_neighbors
    ]

    # ---------- helper function ----------
    def keep_full_shell(neighborlist, k):
        """
        Keep the first k nearest neighbors, but also include all neighbors
        whose distances are tied with the k-th neighbor within tolerance.
        """
        if len(neighborlist) <= k:
            return neighborlist

        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        kth_dist = neighborlist[k - 1][2]

        return [
            nbr for nbr in neighborlist
            if nbr[2] <= kth_dist + tol
        ]

    # ---------- expand cutoff if interlayer neighbors are insufficient ----------
    for site_idx, neighborlist in enumerate(all_neighbors):

        # Only count neighbors from the opposite layer.
        inter = [
            nbr for nbr in neighborlist
            if z_layer[site_idx] != z_layer[nbr[1]]
        ]

        inter_sorted = sorted(inter, key=lambda x: x[2])
        inter_shell = keep_full_shell(inter_sorted, max_neighbors)

        # If any atom has fewer than max_neighbors interlayer neighbors,
        # increase the cutoff and recompute the neighbor list.
        if len(inter_shell) < max_neighbors:
            new_cutoff = 2.0 * cutoff

            if new_cutoff > max_cutoff:
                print("Reached max_cutoff. Stop expanding.")
                return defaultdict(set), defaultdict(set), cutoff

            return nearest_neighbor_edges_vdw_full(
                atoms=atoms,
                cutoff=new_cutoff,
                max_neighbors=max_neighbors,
                z_cut=z_cut,
                id=id,
                use_canonize=use_canonize,
                max_cutoff=max_cutoff,
                tol=tol,
            )

    # ---------- build intralayer and interlayer edges ----------
    intra_edges = defaultdict(set)
    inter_edges = defaultdict(set)

    for site_idx, neighborlist in enumerate(all_neighbors):

        intra = []
        inter = []

        # Split neighbors into intralayer and interlayer groups.
        for nbr in neighborlist:
            dst = nbr[1]

            if z_layer[site_idx] == z_layer[dst]:
                intra.append(nbr)
            else:
                inter.append(nbr)

        # Interlayer edges:
        # Keep only max_neighbors nearest neighbors, including tied neighbors
        # in the same distance shell.
        inter_sorted = sorted(inter, key=lambda x: x[2])
        inter_shell = keep_full_shell(inter_sorted, max_neighbors)

        for nbr in inter_shell:
            dst = nbr[1]
            image = tuple(nbr[3])

            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), image
            )

            if use_canonize:
                inter_edges[(src_id, dst_id)].add(dst_image)
            else:
                inter_edges[(site_idx, dst)].add(image)

        # Intralayer edges:
        # Keep all intralayer neighbors found within the final cutoff.
        for nbr in intra:
            dst = nbr[1]
            image = tuple(nbr[3])

            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), image
            )

            if use_canonize:
                intra_edges[(src_id, dst_id)].add(dst_image)
            else:
                intra_edges[(site_idx, dst)].add(image)

    return intra_edges, inter_edges, cutoff
def nearest_neighbor_edges_coulomb(
    atoms=None,
    cutoff=4.0,
    max_neighbors=12,
    z_cut=None,
    use_canonize=False,
    max_cutoff=100,
    tol=1e-6,
):
    coords = atoms.cart_coords
    z_layer = coords[:, 2] > z_cut
    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    # --------- helper: shell-aware truncate ----------
    def keep_full_shell(neighborlist, k):
        if len(neighborlist) <= k:
            return neighborlist

        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        kth_dist = neighborlist[k - 1][2]

        return [
            nbr for nbr in neighborlist
            if nbr[2] <= kth_dist + tol
        ]

    # --------- check if need expand (shell-aware) ----------
    for site_idx, neighborlist in enumerate(all_neighbors):
       
        intra = [
    nbr
    for nbr in neighborlist
    if (
        nbr[0] == site_idx
        and z_layer[nbr[0]] == z_layer[nbr[1]]
        and nbr[3][2] == 0
        and not (
            nbr[1] == site_idx and tuple(nbr[3]) == (0, 0, 0)
        )
    )
]
        intra_sorted = sorted(intra, key=lambda x: x[2])
        intra_shell = keep_full_shell(intra_sorted, max_neighbors)

        if len(intra_shell) < max_neighbors:
            new_cutoff = 2.0 * cutoff
            if new_cutoff > max_cutoff:
                print("Reached max_cutoff. Stop expanding.")
                return {}, cutoff

            return nearest_neighbor_edges_coulomb(
                atoms=atoms,
                cutoff=new_cutoff,
                max_neighbors=max_neighbors,
                z_cut=z_cut,
                use_canonize=use_canonize,
                max_cutoff=max_cutoff,
                tol=tol,
            )

    # --------- build edges ----------
    edges = defaultdict(set)

    for site_idx, neighborlist in enumerate(all_neighbors):

        intra = [
    nbr
    for nbr in neighborlist
    if (
        nbr[0] == site_idx
        and z_layer[nbr[0]] == z_layer[nbr[1]]
        and nbr[3][2] == 0
        and not (
            nbr[1] == site_idx and tuple(nbr[3]) == (0, 0, 0)
        )
    )
]
        intra_sorted = sorted(intra, key=lambda x: x[2])
        intra_shell = keep_full_shell(intra_sorted, max_neighbors)

        for nbr in intra_shell:
            dst = nbr[1]
            image = tuple(nbr[3])

            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), image
            )

            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(image)
    # extra_imgs_2d = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
    # num_atoms = len(coords)

    # for i in range(num_atoms):
    #     for img in extra_imgs_2d:
    #         if use_canonize:
    #             # canonize self-edge too, to stay consistent
    #             src_id, dst_id, src_image, dst_image = canonize_edge(
    #                 i, i, (0, 0, 0), img
    #             )
    #             edges[(src_id, dst_id)].add(dst_image)
    #         else:
    #             edges[(i, i)].add(img)

    return edges, cutoff





from collections import defaultdict
import numpy as np
from jarvis.core.graphs import canonize_edge
import torch
from collections import defaultdict

def build_line_graph_no_duplicate(u, v, r):
    """
    Build angle list (line graph) without duplicate angles.

    Parameters
    ----------
    u : (E,) source atom indices
    v : (E,) destination atom indices  ← center will be here
    r : (E,3) displacement vectors (src -> dst)

    Returns
    -------
    angle_edge_index : (2, A)
        indices of two edges forming each angle
    angle_attr : (A,)
        cos(theta)
    """

    # Map: center atom → list of incoming edge indices
    center_to_edges = defaultdict(list)
    for edge_id, (src, dst) in enumerate(zip(u.tolist(), v.tolist())):
        center_to_edges[dst].append(edge_id)

    lg_e1 = []
    lg_e2 = []
    angle_attr = []

    # Loop over each center atom
    for center, edge_list in center_to_edges.items():

        n = len(edge_list)
        if n < 2:
            continue

        # choose unique pairs only (i < j)
        for i in range(n):
            for j in range(i + 1, n):

                e1 = edge_list[i]
                e2 = edge_list[j]

                # vectors must point OUTWARD from center
                # current r is (src -> dst)
                # since dst = center, r already points into center
                # so we flip sign to get center -> neighbor
                r1 = r[e1]
                r2 = r[e2]

                cos_theta = torch.dot(r1, r2) / (
                    torch.norm(r1) * torch.norm(r2) + 1e-12
                )

                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

                lg_e1.append(e1)
                lg_e2.append(e2)
                angle_attr.append(cos_theta)

    if len(lg_e1) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=r.dtype),
        )

    angle_edge_index = torch.tensor([lg_e1, lg_e2], dtype=torch.long)
    angle_attr = torch.stack(angle_attr)

    return angle_edge_index, angle_attr

def nearest_neighbor_edges2(
    atoms=None, cutoff=8, max_neighbors=12,
    id=None, use_canonize=False, z_cut=None
):
    """Construct k-NN edge list and count inter/intra layer neighbors."""

    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff

        return nearest_neighbor_edges2(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
            z_cut=z_cut
        )

    edges = defaultdict(set)

    intra_count = 0
    inter_count = 0

    cart_coords = atoms.cart_coords

    # ===== NEW: lưu layer của từng atom =====
    layer_mask = None
    if z_cut is not None:
        z_coords = cart_coords[:, 2]
        layer_mask = (z_coords > z_cut).astype(int)

    # ===== NEW: lưu loại cạnh =====
    edge_layer_type = {}

    for site_idx, neighborlist in enumerate(all_neighbors):

        neighborlist = sorted(neighborlist, key=lambda x: x[2])

        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        max_dist = distances[max_neighbors - 1]

        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        for dst, image in zip(ids, images):
            if image[2] != 0:
                continue
            # ===== ĐẾM inter / intra =====
            if z_cut is not None:
                z1 = cart_coords[site_idx][2]
                z2 = cart_coords[dst][2]

                layer1 = z1 > z_cut
                layer2 = z2 > z_cut

                if layer1 == layer2:
                    intra_count += 1
                    edge_layer_type[(site_idx, dst, tuple(image))] = 0  # intra
                else:
                    inter_count += 1
                    edge_layer_type[(site_idx, dst, tuple(image))] = 1  # inter

            # ===== build edge =====
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )

            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    if z_cut is not None:
        print("Intralayer neighbors:", intra_count)
        print("Interlayer neighbors:", inter_count)
        print("Ratio:", intra_count * 100 / (intra_count + inter_count))

    # ===== NEW: return thêm layer_mask và edge_layer_type =====
    return edges, layer_mask, edge_layer_type

def load_radius_graphs(
        df: pd.DataFrame,
        name: str = "dft_3d",
        target: str = "",
        radius: float = 4.0,
        max_neighbors: int = 40,
        cachedir: Optional[Path] = None,
):
    def atoms_to_graph(atoms):
        structure = Atoms.from_dict(atoms)
        z_cut=atoms["z_cut"]
        sps_features = []
        for ii, s in enumerate(structure.elements):
            feat = list(get_node_attributes(s, atom_features="atomic_number"))
            sps_features.append(feat)
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(torch.get_default_dtype())
        z = torch.tensor(structure.cart_coords[:, 2], dtype=torch.get_default_dtype())
        layer_index = (z > z_cut).long()
        bot_ids = torch.where(layer_index == 0)[0]
        top_ids = torch.where(layer_index == 1)[0]

        species = [str(e) for e in structure.elements]
        bot_species = [species[i] for i in bot_ids.tolist()]
        top_species = [species[i] for i in top_ids.tolist()]

        same_num_atoms = len(bot_species) == len(top_species)
        same_species = sorted(bot_species) == sorted(top_species)

        is_homo_bilayer = bool(same_num_atoms and same_species)
        # ---------- Coulomb edges ----------
        #edges= nearest_neighbor_edges(atoms=structure,cutoff=radius, max_neighbors=max_neighbors*2)
        
        edges,r_cut = nearest_neighbor_edges_coulomb(atoms=structure,cutoff=radius,max_neighbors=max_neighbors,z_cut=z_cut)
        #u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)
        #num_edges = r.shape[0]
    
 
    
        radius_vdw = 4.0
        # ---------- vdW edges ----------
        edges_vdw,r_cut_vdw= nearest_neighbor_edges_vdw(atoms=structure,cutoff=radius_vdw,max_neighbors=max_neighbors,z_cut=z_cut)
        u, v, r = build_undirected_edgedata(atoms=structure, edges=edges)
        u_vdw, v_vdw, r_vdw= build_undirected_edgedata(atoms=structure, edges=edges_vdw)
        num_edges = r.shape[0]
       
        num_vdw_edges = r_vdw.shape[0]  
     
        r_norm_vdw = r_vdw
        edge_vdw_attr = r_norm_vdw

     
        # ---------- pack Data ----------
        data = Data(
            x=node_features,
            # -------- Coulomb --------
            edge_index=torch.stack([u, v]),
            edge_attr=r,

            # -------- vdW edges --------
            edge_vdw_index= torch.stack([u_vdw, v_vdw]),
            r_vdw= r_vdw,
            layer_index=layer_index,
            is_homo_bilayer=torch.tensor([is_homo_bilayer], dtype=torch.bool),
        )
        return data

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{target}-radius.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        pass
    else:
        graphs = df["atoms"].parallel_apply(atoms_to_graph).values
        #graphs = df["atoms"].apply(atoms_to_graph).values
        torch.save(graphs, cachefile)
class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_top":
            return self.x_top.size(0)
        if key == "edge_index_bot":
            return self.x_bot.size(0)
        return super().__inc__(key, value, *args, **kwargs)
def load_radius_graphs2(
        df: pd.DataFrame,
        name: str = "dft_3d",
        target: str = "",
        radius: float = 4.0,
        max_neighbors: int = 40,
        cachedir: Optional[Path] = None,
):
    def mono_atoms_to_graph(atoms):
        structure = Atoms.from_dict(atoms)

        sps_features = []
        for s in structure.elements:
            feat = list(get_node_attributes(s, atom_features="atomic_number"))
            sps_features.append(feat)

        sps_features = np.array(sps_features)
        node_features = torch.tensor(
            sps_features,
            dtype=torch.get_default_dtype()
        )

        edges = nearest_neighbor_edges(
            atoms=structure,
            cutoff=radius,
            max_neighbors=max_neighbors
        )

        u, v, r = build_undirected_edgedata(
            atoms=structure,
            edges=edges
        )

        data = Data(
            x=node_features,
            edge_index=torch.stack([u, v]),
            edge_attr=r,
        )

        return data

    def row_to_two_graphs(row):
        atoms_bot = row["atoms_bot"]
        atoms_top = row["atoms_top"]

        data_bot = mono_atoms_to_graph(atoms_bot)
        data_top = mono_atoms_to_graph(atoms_top)

        data = PairData()

        data.x_bot = data_bot.x
        data.edge_index_bot = data_bot.edge_index
        data.edge_attr_bot = data_bot.edge_attr
        data.num_nodes_bot = data_bot.x.size(0)

        data.x_top = data_top.x
        data.edge_index_top = data_top.edge_index
        data.edge_attr_top = data_top.edge_attr
        data.num_nodes_top = data_top.x.size(0)

        if target != "" and target in row:
            data.y = torch.tensor(
                [row[target]],
                dtype=torch.get_default_dtype()
            )

        return data

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{target}-radius.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs = torch.load(cachefile)
    else:
        graphs = df.apply(row_to_two_graphs, axis=1).values

        if cachefile is not None:
            torch.save(graphs, cachefile)

    return graphs


def get_id_train_val_test(
        total_size=1000,
        split_seed=123,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        n_train=None,
        n_test=None,
        n_val=None,
        keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
            train_ratio is None
            and val_ratio is not None
            and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1

    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)

    if n_train + n_val + n_test > total_size:
        print(n_train ,n_val , n_test)
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-n_test:] #ids[-(n_val + n_test): -n_test]  
    id_test = ids[-n_test:]
    id_val = id_test
    return id_train, id_val, id_test
    #return id_train, id_test

def nearest_neighbor_edges(
    atoms=None, cutoff=8, max_neighbors=12, id=None, use_canonize=False,z_cut=None
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    # if a site has too few neighbors, increase the cutoff radius
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    # print ('cutoff=',all_neighbors)
    if min_nbrs < max_neighbors:
        # print("extending cutoff radius!", attempt, cutoff, id)
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    # build up edge list
    # NOTE: currently there's no guarantee that this creates undirected graphs
    # An undirected solution would build the full edge list where nodes are
    # keyed by (index, image), and ensure each edge has a complementary edge

    # indeed, JVASP-59628 is an example of a calculation where this produces
    # a graph where one site has no incident edges!

    # build an edge dictionary u -> v
    # so later we can run through the dictionary
    # and remove all pairs of edges
    # so what's left is the odd ones out
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):

        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        # max_dist = distances[max_neighbors - 1]

        # keep all edges out to the neighbor shell of the k-th neighbor
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        # keep track of cell-resolved edges
        # to enforce undirected graph construction
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges

def get_torch_dataset(
        dataset=None,
        root="",
        cachedir="",
        processdir="",
        name="",
        id_tag="jid",
        target="",
        atom_features="",
        normalize=False,
        euclidean=False,
        cutoff=4.0,
        max_neighbors=16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
        mean=0.0,
        std=1.0,
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    print(df)
    vals = df[target].values
    print(vals)
    print("data range", np.max(vals), np.min(vals))
    cache = os.path.join(root, cachedir)
    if not os.path.exists(cache):
        os.makedirs(cache)
    if euclidean:
        load_radius_graphs(
            df,
            radius=cutoff,
            max_neighbors=max_neighbors,
            name=name + "-" + str(cutoff),
            target=target,
            cachedir=Path(cache),
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{cutoff}-{target}-radius.bin"),
            processdir,
            target=target,
            name=f"{name}-{cutoff}-{target}-radius",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    else:
        load_infinite_graphs(
            df,
            name=name,
            target=target,
            cachedir=Path(cache),
            infinite_funcs=infinite_funcs,
            infinite_params=infinite_params,
            R=R,
        )

        data = StructureDataset(
            df,
            os.path.join(cachedir, f"{name}-{target}-infinite.bin"),
            processdir,
            target=target,
            name=f"{name}-{target}-infinite",
            atom_features=atom_features,
            id_tag=id_tag,
            root=root,
            mean=mean,
            std=std,
            normalize=normalize,
        )
    return data


def get_train_val_loaders(
        dataset: str = "dft_3d",
        root: str = "",
        cachedir: str = "",
        processdir: str = "",
        dataset_array=None,
        target= None,
        atom_features: str = "cgcnn",
        n_train=None,
        n_val=None,
        n_test=None,
        train_ratio=None,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size: int = 64,
        split_seed: int = 123,
        keep_data_order=False,
        workers: int = 4,
        pin_memory: bool = True,
        id_tag: str = "jid",
        normalize=False,
        euclidean=False,
        cutoff: float = 4.0,
        max_neighbors: int = 16,
        infinite_funcs=[],
        infinite_params=[],
        R=5,
):
    # if not dataset_array:
    #     d = jdata(dataset)
    # else:
    d = dataset_array
    #print(d)

    dat = []
    all_targets = []

    for i in d:
        if isinstance(i[target], list):
            all_targets.append(torch.tensor(i[target]))
            dat.append(i)

        elif (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
        ):
            dat.append(i)
            all_targets.append(i[target])
        #print(i)
    all_targets = torch.tensor(all_targets, dtype=torch.float)  # (N, out_dim)
        

          

    id_train, id_val, id_test = get_id_train_val_test(
        total_size=len(dat),
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        keep_data_order=keep_data_order,
    )
    ids_train_val_test = {}
    ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
    ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
    ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
    dumpjson(
        data=ids_train_val_test,
        filename=os.path.join(root, "ids_train_val_test.json"),
    )
    dataset_train = [dat[x] for x in id_train]
    dataset_val = [dat[x] for x in id_val]
    dataset_test = [dat[x] for x in id_test]
    
    #print(dataset_test)

    # print('using mp bulk dataset')
    # with open('/data/kruskallin/bulk_megnet_train.pkl', 'rb') as f:
    #     dataset_train = pk.load(f)
    # with open('/data/kruskallin/bulk_megnet_val.pkl', 'rb') as f:
    #     dataset_val = pk.load(f)
    # with open('/data/kruskallin/bulk_megnet_test.pkl', 'rb') as f:
    #     dataset_test = pk.load(f)
    #
    # target = 'bulk modulus'

    # print('using mp shear dataset')
    # with open('/data/kruskallin/shear_megnet_train.pkl', 'rb') as f:
    #     dataset_train = pk.load(f)
    # with open('/data/kruskallin/shear_megnet_val.pkl', 'rb') as f:
    #     dataset_val = pk.load(f)
    # with open('/data/kruskallin/shear_megnet_test.pkl', 'rb') as f:
    #     dataset_test = pk.load(f)
    # target = 'shear modulus'

    start = time.time()
    train_data = get_torch_dataset(
        dataset=dataset_train,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_train",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
    )

    mean = train_data.mean
    std = train_data.std

    val_data = get_torch_dataset(
        dataset=dataset_val,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_val",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
        mean=mean,
        std=std
    )

    test_data = get_torch_dataset(
        dataset=dataset_val,
        root=root,
        cachedir=cachedir,
        processdir=processdir,
        name=dataset + "_test",
        id_tag=id_tag,
        target=target,
        atom_features=atom_features,
        normalize=normalize,
        euclidean=euclidean,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        infinite_funcs=infinite_funcs,
        infinite_params=infinite_params,
        R=R,
        mean=mean,
        std=std)
    # ) get_torch_dataset(
    #     dataset=dataset_test,
    #     root=root,
    #     cachedir=cachedir,
    #     processdir=processdir,
    #     name=dataset + "_test",
    #     id_tag=id_tag,
    #     target=target,
    #     atom_features=atom_features,
    #     normalize=normalize,
    #     euclidean=euclidean,
    #     cutoff=cutoff,
    #     max_neighbors=max_neighbors,
    #     infinite_funcs=infinite_funcs,
    #     infinite_params=infinite_params,
    #     R=R,
    #     mean=mean,
    #     std=std,
    # )
    print(test_data)
    print("------processing time------: " + str(time.time() - start))

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
        follow_batch=["x_top", "x_bot"],
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
        follow_batch=["x_top", "x_bot"],
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
        follow_batch=["x_top", "x_bot"],
    )

    print("n_train:", len(train_loader.dataset))
    # print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        mean,
        std,
    )
