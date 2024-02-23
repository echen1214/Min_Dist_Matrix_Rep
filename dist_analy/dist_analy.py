"""Main module."""

import warnings
from pathlib import Path
from typing import List, Optional

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

    # if not isinstance(residues1, ndarray):
    #     raise TypeError('residues1 must be an array')
    # if not isinstance(residues1[0], Residue):
    #     raise TypeError('array must contain Residue objects')

    atomic_xyz_residue_1 = get_atom_coords(heavy, residues1, Residue)

    if residues2 is None or np.array_equal(res_list_2,res_list_1):
        # print("here")
        symmetric = True
        residues2 = residues1
        res_list_2 = res_list_1
        atomic_xyz_residue_2 = atomic_xyz_residue_1
    else:
        symmetric = False
        no_adj = False
        if mol:
            check = AtomGroup
        else:
            check = Residue

        atomic_xyz_residue_2 = get_atom_coords(heavy, residues2, check)

        # if not isinstance(residues2, ndarray):
        #     raise TypeError('residues2 must be an array')
        # if not isinstance(residues2[0], Residue):
        #     raise TypeError('array must contain Residue objects')

        # print(atomcoords1)
    len1 = len(residues1)
    len2 = len(residues2)

    if unitcell is not None:
        if not isinstance(unitcell, ndarray):
            raise TypeError('unitcell must be an array')
        elif unitcell.shape != (3,):
            raise ValueError('unitcell.shape must be (3,)')

    if format not in DISTMAT_FORMATS:
        raise ValueError('format must be one of mat, rcd, or arr')

    if format == 'mat':
        dist = zeros((len1, len2))
    else:
        dist = []

    if no_adj:
        if symmetric:
            no_adj_res = [res.select('not backbone') if isinstance(res, Residue) else None for res in residues1]
            no_adj_res_coords = empty(len1, dtype=object)
            no_adj_res_truthy = [False] * len1
            for i, x in enumerate(no_adj_res):
                if x:
                    no_adj_res_coords[i] = x.getCoords()
                    no_adj_res_truthy[i] = True

    if symmetric:
        for i, atoms_group_res1 in enumerate(atomic_xyz_residue_1[:-1]):
            for j, atoms_group_res2 in enumerate(atomic_xyz_residue_2[i + 1:]):
                j_1 = i + j + 1 # This is what, a pointer to the next residue? Why not just use j?
                # Residue Block
                if atoms_group_res1 is None or atoms_group_res2 is None:
                    value = 0
                else:
                    res1_t = atoms_group_res1
                    if no_adj and abs(res_list_2[j_1] - res_list_1[i]) == 1:
                        if no_adj_res_truthy[i] and no_adj_res_truthy[j_1]:
                            res1_t = no_adj_res_coords[i]
                            atoms_group_res2 = no_adj_res_coords[j_1]
                        else:
                            res1_t = [np.array([5.0, 0.0, 0.0])]
                            atoms_group_res2 = [np.array([0.0, 0.0, 0.0])]

                    if unitcell is not None: # Legacy
                        temp_dist = []
                        for atom1 in res1_t:
                            temp_dist.append(getDistance(atom1, atoms_group_res2, unitcell))
                        value = np.min(temp_dist)
                    else:
                        value = np.min(distance_matrix(res1_t, atoms_group_res2))
                    if min_dist and value < min_dist:
                        value = min_dist


                if format == 'mat':
                    dist[i, j_1] = dist[j_1, i] = value
                else:
                    dist.append(value)
        if format == 'rcd':
            n_res1 = len(residues1)
            n_res2 = len(residues2)
            rc = array([(i, j) for i in range(n_res1) \
                        for j in range(i + 1, n_res2)])
            row, col = rc.T
            dist = (row, col, dist)

    else:
        for i, atoms_group_res1 in enumerate(atomic_xyz_residue_1):
            for j, atoms_group_res2 in enumerate(atomic_xyz_residue_2):
                if atoms_group_res1 is None or atoms_group_res2 is None:
                    value = 0
                else:
                    res1_t = atoms_group_res1
                    res2_t = atoms_group_res2
                    if no_adj and abs(res_list_2[j] - res_list_1[i]) == 1:
                        atom1_noadj = residues1[i].select('not backbone')
                        atom2_noadj = residues2[j].select('not backbone')
                        if atom1_noadj and atom2_noadj:
                            res1_t = np.array([x.getCoords() for x in atom1_noadj])
                            res2_t = np.array([x.getCoords() for x in atom2_noadj])
                        else:
                            res1_t = [np.array([5.0, 0.0, 0.0])]
                            res2_t = [np.array([0.0, 0.0, 0.0])]

                    if unitcell is not None: #Legacy
                        temp_dist = []
                        for atom1 in res1_t:
                            for atom2 in res2_t:
                                temp_dist.append(getDistance(atom1, atom2, unitcell))

                        value = np.min(temp_dist)

                    else:
                        value = np.min(distance_matrix(res1_t, res2_t))
                    if min_dist and value < min_dist:
                        value = min_dist
                if format == 'mat':
                    dist[i, j] = value
                else:
                    dist.append(value)
        if format == 'rcd':
            n_res1 = len(residues1)
            n_res2 = len(residues2)
            rc = np.array([(i, j) for i in range(n_res1)
                           for j in range(n_res2)])
            row, col = rc.T
            dist = (row, col, dist)

    return dist


def get_atom_coords(heavy: bool, residues1: List, type):
    if heavy:
        atomcoords1 = np.array([x.select('heavy').getCoords() if isinstance(x, type) else None for x in residues1],
                               dtype=ndarray)
    else:
        atomcoords1 = np.array([x.getCoords() if isinstance(x, type) else None for x in residues1], dtype=ndarray)
    return atomcoords1

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

def handle_ligand_files(file: str, ligand_file: str, ligand_chain: str, 
                        save_dir: str, save_fn: str, *args, **kwargs):
    """ handle whether or not a single ligand file or ligand list will be
    calculated

    Parameters
    ----------
    file : str
        receptor file name
    ligand_file : str | list
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
    if isinstance(ligand_file, list):
        dist_matrix = []
        for ligand in ligand_file:
            dist_matrix_1 = build_shortest_receptor_ligand_matrix(file, ligand, ligand_chain, save_dir, save_fn, *args, **kwargs)
            dist_matrix.append(dist_matrix_1)
        dist_matrix = np.array(dist_matrix)
    elif isinstance(ligand_file, str):
        dist_matrix = build_shortest_receptor_ligand_matrix(file, ligand_file, ligand_chain, save_dir, save_fn, *args, **kwargs)

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

def build_shortest_receptor_ligand_matrix(file: str, ligand_file:str|list, ligand_chain:str, \
                                    save_dir: str, save_fn: str, *args, **kwargs):
    """ Helper function to calculate ligand distance matrix
    
    passing in extra variables and extra keyword arguments will be passed
    to `build_shortest_dist_matrix`
    
    Parameters
    ----------
    file : str
        receptor file name
    ligand_file : str | list
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
    ligand = pdbfile.parsePDB(ligand_file, chain=ligand_chain)
    if ligand.numResidues() > 1:
        warnings.warn("Ligand is more than one residue")
    lig_res = list(set(ligand.getResnums()))

    dist_matrix = build_shortest_dist_matrix(*args, residues2=[ligand], res_list_2=lig_res, **kwargs).T[0]  
    
    return dist_matrix

def get_shortest_dist_matrix(file: str, res_list: list = None, chain: str = None,
                             min_dist: int = None, no_adj: bool = True,
                             save_dir: str = None, save_fn: str = None, ligand_file: str|list = None,
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
