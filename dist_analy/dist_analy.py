"""Main module."""

import numpy as np
from numpy import zeros, array, ndarray, empty, any, save
from prody.atomic import Atomic, Residue, Atom, AtomGroup
from prody.proteins import pdbfile
from prody.measure import measure
from prody.utilities import getDistance
from pathlib import Path

""" TODO
    - x separate the pdb files into multiple models (multiple pdb files for
        each chain and each model with varying occupancy)
    - x renumber based on SIFTs numbering'
    - option to not remove residues and get the missing data to median/mean

    - implement reading of other types of file (PQR, mol2, CIF)
"""

DISTMAT_FORMATS = set(['mat', 'rcd', 'arr'])

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

    ca_atoms = atoms.select('resnum %s'%(" ".join([str(x) for x in res_list])))
    # print(ca_atoms)
    dist_matrix_sel = measure.buildDistMatrix(ca_atoms)
    # print(dist_matrix_sel)
    res_truthy = np.zeros(len(res_list), dtype=bool)
    reindex = np.zeros(len(ca_atoms), dtype = int)
    for i,atom in enumerate(ca_atoms):
        res_truthy[res_list.index(atom.getResnum())] = True
        reindex[i] = res_list.index(atom.getResnum())
    # print(res_truthy)
    # print(reindex, type(reindex[0]))

    if not len(dist_matrix_sel) == np.count_nonzero(res_truthy):
        raise ValueError("residue selection went wrong")

    dist_matrix = np.zeros((len(res_list),len(res_list)))
    for i,row in enumerate(dist_matrix_sel):
        for j,val in enumerate(row):
            dist_matrix[reindex[i]][reindex[j]] = val
    # print(dist_matrix)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fn = file.split('/')[-1].split('.')[0]
        np.save(save_dir+fn, dist_matrix)

    return dist_matrix

def get_shortest_dist(res1: ndarray, res2: ndarray, unitcell:ndarray=None, min_dist: int = None):
    """Helper function for build_shortest_dist_matrix.
    Loops through pairwise residue-residue distances and returns the shortest

    Parameters
    ----------
    res1 : ndarray
        array of atomic coordinates
    res2 : ndarray
        array of atomic coordinates
    unitcell : numpy.ndarray, optional
        orthorhombic unitcell dimension array with shape ``(3,)``
    min_dist : int, optional
        if true the minimum residue-residue distance is set to min_dist

    Returns
    -------
    np.float32
        value of the shortest distance between two residues

    """
    temp_dist = []
    for atom1 in res1:
        for atom2 in res2:
            temp_dist.append(getDistance(atom1, atom2, unitcell))
    value = np.min(temp_dist)

    if min_dist and value < min_dist:
        value = min_dist

    return value


def build_shortest_dist_matrix(residues1:ndarray, res_list_1:ndarray, residues2:ndarray=None,
                              res_list_2:ndarray=None, unitcell:ndarray=None,
                              format='mat', no_adj: bool = True, min_dist: int = None):
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
    residues2 : prody.atomic.Residue objects in np.ndarray, optional
        residue or coordinate data
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

    Returns
    -------
    np.array
        When *residues2* is given, a distance matrix with shape ``(len(residues1),
        len(residues2))`` is built.  When *residues2* is **None**, a symmetric matrix
        with shape ``(len(residues1), len(residues1))`` is built.  If *unitcell* array
        is provided, periodic boundary conditions will be taken into account.

    """

    if not isinstance(residues1, ndarray):
        raise TypeError('residues1 must be an array')
    # if not isinstance(residues1[0], Residue):
    #     raise TypeError('array must contain Residue objects')

    atomcoords1 = np.array([x.getCoords() if isinstance(x,Residue) else None for x in residues1], dtype=ndarray)

    if residues2 is None:
        symmetric = True
        residues2 = residues1
        res_list_2 = res_list_1
        atomcoords2 = atomcoords1
    else:
        symmetric = False
        atomcoords1 = np.array([x.getCoords() if isinstance(x,Residue) else None for x in residues2], dtype=ndarray)

        if not isinstance(residues2, ndarray):
            raise TypeError('residues2 must be an array')
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
            no_adj_res = [res.select('not backbone') if isinstance(res,Residue) else None for res in residues1]
            no_adj_res_coords = empty(len1, dtype=object)
            no_adj_res_truthy = [False] * len1
            for i,x in enumerate(no_adj_res):
                if x:
                    no_adj_res_coords[i] = x.getCoords()
                    no_adj_res_truthy[i] = True

    if symmetric:
        for i,res1 in enumerate(atomcoords1[:-1]):
            for j,res2 in enumerate(atomcoords2[i+1:]):
                j_1 = i + j + 1
                if res1 is None or res2 is None:
                    value = 0
                else:
                    res1_t = res1
                    if no_adj and abs(res_list_2[j_1]-res_list_1[i])==1:
                        if no_adj_res_truthy[i] and no_adj_res_truthy[j_1]:
                            res1_t = no_adj_res_coords[i]
                            res2 = no_adj_res_coords[j_1]
                        else:
                            res1_t = [np.array([5.0,0.0,0.0])]
                            res2 = [np.array([0.0,0.0,0.0])]

                    temp_dist = []
                    for atom1 in res1_t:
                        temp_dist.append(getDistance(atom1, res2, unitcell))
                    value = np.min(temp_dist)

                    if min_dist and value < min_dist:
                        value = min_dist

                if format == 'mat':
                    dist[i,j_1] = dist[j_1,i] = value
                else: dist.append(value)
        if format == 'rcd':
            n_res1 = len(residues1)
            n_res2 = len(residues2)
            rc = array([(i, j) for i in range(n_res1) \
                        for j in range(i + 1, n_res2)])
            row, col = rc.T
            dist = (row, col, dist)

    else:
        for i,res1 in enumerate(atomcoords1):
            for j,res2 in enumerate(atomcoords2):
                if res1 is None or res2 is None:
                    value = 0
                else:
                    res1_t = res1
                    res2_t = res2
                    temp_dist = []
                    if no_adj and res_list_2[j]-res_list_1[i]==1:
                        atom1_noadj = residues1[i].select('not backbone')
                        atom2_noadj = residues2[j].select('not backbone')
                        if atom1_noadj and atom2_noadj:
                            res1_t = np.array([x.getCoords() for x in atom1_noadj])
                            res2_t = np.array([x.getCoords() for x in atom2_noadj])
                        else:
                            res1_t = [np.array([5.0,0.0,0.0])]
                            res2_t = [np.array([0.0,0.0,0.0])]

                    # value = get_shortest_dist(res1_t, res2_t, unitcell, min_dist)
                    for atom1 in res1_t:
                        for atom2 in res2_t:
                            temp_dist.append(getDistance(atom1, atom2, unitcell))

                    value = np.min(temp_dist)
                    if min_dist and value < min_dist:
                        value = min_dist

                    if format == 'mat':
                        dist[i,j] = value
                    else: dist.append(value)
        if format == 'rcd':
            n_res1 = len(residues1)
            n_res2 = len(residues2)
            rc = np.array([(i, j) for i in range(n_res1) \
                            for j in range(n_res2)])
            row, col = rc.T
            dist = (row, col, dist)

    return dist


def get_shortest_dist_matrix(file: str, res_list: list, chain: str,
                             min_dist: int = None, no_adj: bool = True,
                             save_dir: str = None):
    """ Generate shortest-distance distance matrix.

    Parameters
    ----------
    file : str
        file name
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

    Returns
    -------
    np.array
        2D np.array of carbon alpha distance matrix

    """
    structure= pdbfile.parsePDB(file, chain=chain)
    hv = structure.getHierView()
    obj = hv[chain]
    res_obj = np.empty(len(res_list), dtype=Residue)
    for i,res in enumerate(res_list):
        temp_obj = obj.getResidue(res)
        if temp_obj:
            res_obj[i] = temp_obj
    # res_obj= np.array(hv[chain], dtype=Residue)[res_list]
    dist_matrix = build_shortest_dist_matrix(res_obj, res_list, min_dist=min_dist, \
                                            no_adj=no_adj)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fn = file.split('/')[-1].split('.')[0]
        np.save(save_dir+fn, dist_matrix)
    return dist_matrix
