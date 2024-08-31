"""Main module."""

import warnings
from pathlib import Path
from typing import List, Optional

import time
import numpy as np
from numpy import array, empty, ndarray, zeros
from prody.atomic import AtomGroup, Residue
from prody.measure import measure
from prody.proteins import pdbfile
from prody.utilities import getDistance
from scipy.spatial import distance_matrix

""" TODO
    - x separate the pdb files into multiple models (multiple pdb files for
        each chain and each model with varying occupancy)
    - x renumber based on SIFTs numbering'
    - x  option to not remove residues and get the missing data to median/mean
    - turn distance matrix into a data structure ?
    - implement reading of other types of file (PQR, mol2, CIF)
    - implement ability to track the atoms pairs involved in the shortest residue
      distances
"""

DISTMAT_FORMATS = set(['mat', 'rcd', 'arr'])

# def get_contact_map(file: str, res_list: list, chain: str, save_dir: str = None):
#     """ Create a contact map following https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0226702#sec008

#     Parameters
#     ----------
#     file : str
#         filename
#     res_list : list
#         list of residues to calculate distance matrix with
#     chain: str
#         chain ID
#     save_dir: str, optional
#         directory to save distance matrices to a binary file in NumPy .npy format
#     """    

def get_ca_dist_matrix(file: str, res_list: list, chain: str, save_dir: str = None):
    """ Using prody functions to generate carbon alpha distance matrix

    Parameters
    ----------
    file : str
        filename
    res_list : list
        list of residues to calculate distance matrix with
    chain: str
        chain ID
    save_dir: str, optional
        directory to save distance matrices to a binary file in NumPy .npy format

    Returns
    -------
    np.array
        2D carbon alpha distance matrix

    """
    # print('ca %s'%(" ".join([str(x) for x in res_list])))
    atoms = pdbfile.parsePDB(file, subset='ca', chain=chain)
    # ca_atoms = atoms[res_list]
    # print(type(ca_atoms), type(ca_atoms[0])) #, ca_atoms.shape[-1])
    # dist_matrix_sel = measure.buildDistMatrix(ca_atoms)

    # ca_atoms = np.empty(len(res_list), dtype=Atom)
    # ca_atoms = []
    # for i,res in enumerate(res_list):
    #     print(atoms.select('resnum %i'%res))
    #     if atoms.select('resnum %i'%res):
    #         ca_atoms.append(atoms.select('resnum %i'%res))

    ca_atoms = atoms.select('resnum %s' % (" ".join([str(x) for x in res_list])))
    # print(ca_atoms)
    dist_matrix_sel = measure.buildDistMatrix(ca_atoms)
    # print(dist_matrix_sel)
    res_truthy = np.zeros(len(res_list), dtype=bool)
    reindex = np.zeros(len(ca_atoms), dtype=int)
    for i, atom in enumerate(ca_atoms):
        res_truthy[res_list.index(atom.getResnum())] = True
        reindex[i] = res_list.index(atom.getResnum())
    # print(res_truthy)
    # print(reindex, type(reindex[0]))

    if not len(dist_matrix_sel) == np.count_nonzero(res_truthy):
        raise ValueError("residue selection went wrong")

    dist_matrix = np.zeros((len(res_list), len(res_list)))
    for i, row in enumerate(dist_matrix_sel):
        for j, val in enumerate(row):
            dist_matrix[reindex[i]][reindex[j]] = val
    # print(dist_matrix)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fn = file.split('/')[-1].split('.')[0]
        np.save(save_dir + fn, dist_matrix)

    return dist_matrix


def parse_residue_list(residue_list, residue_type, heavy, residue_sizes, x_coords, y_coords, z_coords, needs_backbone_sizes = False, backbone_sizes = None):
    for residue in residue_list:
        if not residue:
            residue_sizes.append(0)
            if needs_backbone_sizes:
               backbone_sizes.append(0)
        if not isinstance(residue, residue_type):
            continue
        atoms = []
        if heavy:
            atoms = residue.select('heavy')
        else:
            atoms = residue

        coords = []

        backbone_atoms = atoms.select('backbone')
        if backbone_atoms:
            coords += list(backbone_atoms.getCoords())
            if needs_backbone_sizes:
                backbone_sizes.append(len(coords))

        atoms = atoms.select('not backbone')
        if atoms:
            coords += list(atoms.getCoords())

        # TODO: The repeated array appends might actually be expensive. Investigate pre-allocating these.
        residue_sizes.append(len(coords))
        for coord in coords:
            x_coords.append(coord[0])
            y_coords.append(coord[1])
            z_coords.append(coord[2])


def build_shortest_dist_matrix(residues1: ndarray, res_list_1: list, residues2: ndarray = None,
                               res_list_2: list = None, unitcell: ndarray = None,
                               format='mat', no_adj: bool = True, min_dist: int = None,
                               heavy: bool = True, mol: bool = False):
    """Generate shortest-distance distance matrix
    This code is adapted from the ProDy python package, specifically the
    buildDistMatrix function, under the MIT license

    If the no_adj flag is used and the distance is being calculated between any
    residue and a glycine (or have no side chain), the distance default is set
    to 5 Angstrom

    Parameters
    ----------
    residues1 : prody.atomic.Residue objects in np.ndarray
        residue or coordinate data
    res_list_1 : list
        list of labels of residue1
    residues2 : prody.atomic.Residue objects in np.ndarray, optional
        residue or coordinate data
    res_list_2 : list, optional
        list of labels of residue2
    unitcell : numpy.ndarray, optional
        orthorhombic unitcell dimension array with shape ``(3,)``
    format : bool, default: 'mat'
        format of the resulting array, one of ``'mat'`` (matrix,
        default), ``'rcd'`` (arrays of row indices, column indices, and
        distances), or ``'arr'`` (only array of distances)
    no_adj : bool, default: True
        if true excludes backbone-backbone distances from neighboring residues
    min_dist : int, optional
        if true the minimum residue-residue distance is set to min_dist
    heavy : bool, default: True
        if true, includes only heavy atoms
    mol : bool, default: False
        if true, residue2 is an molecule of type AtomGroup

    Returns
    -------
    np.array
        When *residues2* is given, a distance matrix with shape ``(len(residues1),
        len(residues2))`` is built.  When *residues2* is **None**, a symmetric matrix
        with shape ``(len(residues1), len(residues1))`` is built.  If *unitcell* array
        is provided, periodic boundary conditions will be taken into account.

    """
    is_symmetric = not residues2

    # Parse coordinates from our residue objects.
    residue_indices = res_list_1
    residue_sizes1 = []
    backbone_sizes = []
    x_coords1 = []
    y_coords1 = []
    z_coords1 = []
    residue_sizes2 = []
    x_coords2 = []
    y_coords2 = []
    z_coords2 = []
    parse_residue_list(residues1, Residue, heavy, residue_sizes1, x_coords1, y_coords1, z_coords1, True, backbone_sizes)
    if residues2:
        parse_residue_list(residues2, AtomGroup if mol else Residue, heavy, residue_sizes2, x_coords2, y_coords2, z_coords2, False, None)
    else:
        residue_sizes2 = residue_sizes1
        x_coords2 = x_coords1
        y_coords2 = y_coords1
        z_coords2 = z_coords1

    # Vectorize our coordinates.
    x_coords1 = np.array(x_coords1)
    y_coords1 = np.array(y_coords1)
    z_coords1 = np.array(z_coords1)
    if residues2:
        x_coords2 = np.array(x_coords2)
        y_coords2 = np.array(y_coords2)
        z_coords2 = np.array(z_coords2)
    else:
        x_coords2 = x_coords1
        y_coords2 = y_coords1
        z_coords2 = z_coords1
    x_coords1 = np.tile(x_coords1, (x_coords2.shape[0], 1))
    y_coords1 = np.tile(y_coords1, (y_coords2.shape[0], 1))
    z_coords1 = np.tile(z_coords1, (z_coords2.shape[0], 1))
    if residues2:
        x_coords2 = np.tile(x_coords2, (x_coords1.shape[1], 1))
        y_coords2 = np.tile(y_coords2, (y_coords1.shape[1], 1))
        z_coords2 = np.tile(z_coords2, (z_coords1.shape[1], 1))
    else:
        x_coords2 = x_coords1
        y_coords2 = y_coords1
        z_coords2 = z_coords1       

    # Compute squared distance matrix.
    # TODO: See if there's a numpy function to leverage that will compute the L2 norm in fewer operations.
    # TODO: In the symmetric case, this computes both halves of the matrix when it only needs to compute one.
    dist_matrix = np.square(x_coords1 - np.transpose(x_coords2))
    dist_matrix += np.square(y_coords1 - np.transpose(y_coords2))
    dist_matrix += np.square(z_coords1 - np.transpose(z_coords2))

    # Perform no_adj logic, where we omit calculations between backbones of adjacent residues.
    # This logic is unfortunately fairly scalar, but luckily it only applies to a fraction of the matrix, so it's pretty cheap.
    if not residues2 and no_adj:
        x = residue_sizes1[0]
        y = 0
        for i in range(1, len(residue_sizes1)):
            if not residue_sizes1[i]:
                continue
            next_x = x + residue_sizes1[i]
            next_y = x
            # Just because two residues are adjacent in the residue data does
            # not mean they are actually adjacent in the structure. We need to
            # double check the residue indices before continuing to no_adj
            # logic.
            if residue_indices[i] == residue_indices[i-1] + 1:
                # Glycine is exceptional, because it's all backbone. We just set all distances to 5 in this case
                if residue_sizes1[i-1] == backbone_sizes[i-1] or residue_sizes1[i] == backbone_sizes[i]:
                    dist_matrix[y:next_y, x:next_x] = 25.0 # The dist matrix is still squared.
                else:
                    dist_matrix[y:next_y, x:x+backbone_sizes[i-1]] = 90000.0 # Arbitrarily large number so min won't pick it
                    dist_matrix[y:y+backbone_sizes[i], x+backbone_sizes[i-1]:next_x] = 90000.0
            x = next_x
            y = next_y

    # Find shortest distances within each residue.
    shortest_dist_matrix = np.zeros((len(residue_sizes2), len(residue_sizes1)))
    y_idx = 0
    base_x_idx = 0
    if min_dist == None:
        min_dist = 0
    else:
        min_dist = min_dist**2 # The dist matrix is still squared.
    for i in range(0, len(residue_sizes2)):
        x_idx = 0
        j_start = 0
        if is_symmetric:
            # This matrix is symmetric, so we only calculate the upper right triangle.
            x_idx = y_idx + residue_sizes1[i]
            j_start = i + 1
        for j in range(j_start, len(residue_sizes1)):
            if not residue_sizes2[i] or not residue_sizes1[j]:
                # This protein is missing residues, so just set the corresponding matrix elements to 0 for now.
                # TODO: This might be causing branch mispredictions. Try to find a way to leverage the "initial" param for np.min.
                shortest_dist_matrix[i, j] = 0
            else:
                # TODO: The scalar "max()" call is less efficient than np.max, but we cannot replace missing residue 0s with min_dist.
                # Maybe if we reach the end of the row without missing residues we should call np.max on the whole row?
                shortest_dist_matrix[i, j] = max(np.min(dist_matrix[y_idx:y_idx+residue_sizes2[i], x_idx:x_idx+residue_sizes1[j]]), min_dist)
            x_idx += residue_sizes1[j]
        y_idx += residue_sizes2[i]

    # Take the square root.
    # Note that we do this after the min logic to cut down on the number of square roots we need to calculate.
    shortest_dist_matrix = np.power(shortest_dist_matrix, 0.5)

    # Fill in the other half of the matrix if symmetric.
    # Normally we'd need to divide the diagonal by 2, but since this is a distance matrix, the diagonal should always be 0.
    if is_symmetric:
        shortest_dist_matrix += np.transpose(shortest_dist_matrix)

    return shortest_dist_matrix

def get_res_obj(file: str, chain: str=None, res_list: list=None):
    """ return list of `prody.Residue` objects corresponding the selected
    chain and list of residue numbers

    if no chain is passed then the `prody.Residue` objects corresponding
    to entire file is returned

    Parameters
    ----------
    file : str
        pdb file name
    chain : str, optional
        chain, by default None
    res_list :list, optional
       list of residues, by default None

    Returns
    -------
    list
        list of `prody.Residue` objects
    """    
    structure = pdbfile.parsePDB(file, chain=chain)
    hv = structure.getHierView()

    if chain:
        obj = hv[chain]
    else:
        obj = structure

    if res_list is not None:
        res_obj = np.empty(len(res_list), dtype=Residue)
        for i, res in enumerate(res_list):
            temp_obj = obj.getResidue(res)
            if temp_obj:
                res_obj[i] = temp_obj
    else:
        res_obj = []
        for x in obj:
            res_obj.append(x)

    return (res_obj)

def handle_ligand_files(file: str, ligand_files: str, ligand_chain: str, 
                        save_dir: str, save_fn: str, *args, **kwargs):
    """ handle whether or not a single ligand file or ligand list will be
    calculated

    Parameters
    ----------
    file : str
        receptor file name
    ligand_files : str | list
        ligand file name or list
    ligand_chain : str
        ligand chain
    save_dir : str
        directory to save .npy file in 
    save_fn : str
        path to save .npy file in 

    Returns
    -------
    np.array
        shortest receptor–ligand distance vector
        if a single file is passed, 1D np.array
        if a list of files is passed, 2D np.array
    """
    if isinstance(ligand_files, str):
        ligand_files = [ligand_files]
    ligands = []
    ligand_list = []
    for ligand_file in ligand_files:
        ligands.append(pdbfile.parsePDB(ligand_file, chain=ligand_chain))
        if ligands[-1].numResidues() > 1:
            warnings.warn("Ligand is more than one residue")
        ligand_list += list(set(ligands[-1].getResnums()))

    dist_matrix = build_shortest_dist_matrix(*args, residues2=ligands, res_list_2=ligand_list, **kwargs)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        res_name = list(set(ligand.getResnames()))
        fn = file.split('/')[-1].split('.')[0]
        rank = ligand_file.split('/')[-1].split('.')[0].split('_')[0]
        if len(res_name) == len(lig_res):
            fn = fn + "_".join(["%s%i" % (name, int) for name, int in zip(res_name, lig_res)])
        else:
            fn = fn + "".join(res_name) + "".join([str(i) for i in lig_res])
        # print(fn)
        np.save(save_dir + fn, dist_matrix)
    if save_fn:
        np.save(save_fn, dist_matrix)   

    return dist_matrix

def get_shortest_dist_matrix(file: str, res_list: list = None, chain: str = None,
                             min_dist: int = None, no_adj: bool = True,
                             save_dir: str = None, save_fn: str = None, ligand_file: str = None,
                             ligand_chain: str = None):
    """ Generate shortest-distance distance matrix.

    `save_dir` and `save_fn` are mutually exclusive

    Parameters
    ----------
    file : str
        receptor file name
    res_list : list
        list of residues to calculate distance matrix with
    chain: str
        chain ID
    no_adj : bool, default: True
        if true does not calculate distance between adjacent backbone atoms
    min_dist : int, optional
        if calculated distance is less than this value, set distance to this
        value.
    save_dir: str, optional
        directory to save distance matrices to a binary file in NumPy .npy format
    save_fn: str, optional
        file to save distance matrices to a binary file in NumPy .npy format
    ligand_file: list or str, optional
        pass in a list or single string of ligand files to calculate ligand-receptor
        shortest distance vector with respect to file
    ligand_chain: str, optional
        optional parameter that designating the ligand chain, if none passed and 
        ligand_file pass, the entire ligand_file will be accepted by default

    Returns
    -------
    np.array
        2D np.array of shortest distance matrix

    """
    if save_dir and save_fn:
        raise ValueError("Cannot have save_dir and save_fn at the same time")

    res_obj = get_res_obj(file, chain, res_list)

    if ligand_file:
        dist_matrix = handle_ligand_files(file, ligand_file, ligand_chain, save_dir, save_fn, res_obj, res_list, \
                                         min_dist=min_dist, no_adj=no_adj, heavy=True, mol=True)

    else:
        dist_matrix = build_shortest_dist_matrix(res_obj, res_list, min_dist=min_dist, \
                                        no_adj=no_adj, heavy=True)
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fn = file.split('/')[-1].split('.')[0]
            np.save(save_dir + fn, dist_matrix)
        if save_fn:
            np.save(save_fn, dist_matrix)                                 
    return dist_matrix
