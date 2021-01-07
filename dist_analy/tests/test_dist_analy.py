#!/usr/bin/env python

"""Tests for `dist_analy` package.
- test for multiple chains (reading 1 pdb file with 2 chains of the uniprot protein
    should result in two different structures and distance matrices)
- test for mismatching sequence numbering
- test for multiple occupancies/models
"""

import pytest
import numpy as np
from prody.proteins import pdbfile
from prody.atomic import Residue
from dist_analy import dist_analy

# from dist_analy.dist_analy.tests.datafiles import *

CDK2_KLIFS_IDENT = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 33, 34, 35, 43, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 114, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 139, 142, 143, 144, 145, 146, 147, 148, 149, 163, 165, 167, 168, 169, 171, 174, 185, 187, 190, 192, 195, 220, 227, 262, 274]
MISSING = [37,38,39,40,41,42]

@pytest.mark.parametrize('res_list', [CDK2_KLIFS_IDENT]) # , MISSING])
def test_get_ca_dist_matrix(res_list):
    """Tests get_ca_dist_matrix
    """
    file = './datafiles/pdb_files/testing/4EOQ.pdb'
    chain = 'A'
    mat = dist_analy.get_ca_dist_matrix(file, res_list, chain)
    dist_mat = np.around(mat, 5)

    KI_npy_file = './datafiles/npy_files/ref/4EOQ.pdb_KI.npy'
    KI = np.around(np.load(KI_npy_file), 5)

    assert len(dist_mat) == len(KI)
    assert np.array_equal(dist_mat, KI)
    # print(mat)

@pytest.fixture
def res_obj(res_list):
    """pytest fixture initializing a list of residue objects from the pdb structure
    4EOQ

    Parameters
    ----------
    res_list : list
        list of residues

    Returns
    -------
    np.array
        List of residue objects

    """
    file = './datafiles/pdb_files/testing/4EOQ.pdb'
    chain = 'A'
    structure= pdbfile.parsePDB(file, chain=chain)
    hv = structure.getHierView()
    obj = hv[chain]
    res_obj_list = np.empty(len(res_list), dtype=Residue)
    for i,res in enumerate(res_list):
        if obj.getResidue(res):
            res_obj_list[i] = obj.getResidue(res)
    return res_obj_list

# @pytest.mark.parametrize('res_list', [range(8,11)])
# def test_output(res_obj, res_list):
#     print(dist_analy.build_shortest_dist_matrix(res_obj, res_list, no_adj=False))
#     print(dist_analy.build_shortest_dist_matrix(res_obj, res_list, no_adj=True))

@pytest.mark.parametrize('res_list', [range(3,25)])
def test_flags(res_obj, res_list):
    # print(dist_analy.build_shortest_dist_matrix(res_obj, res_list))
    assert np.any(dist_analy.build_shortest_dist_matrix(res_obj, res_list, no_adj = True) != \
        dist_analy.build_shortest_dist_matrix(res_obj, res_list, no_adj = False))

    assert np.any(dist_analy.build_shortest_dist_matrix(res_obj, res_list, min_dist = 2.2) != \
        dist_analy.build_shortest_dist_matrix(res_obj, res_list, min_dist = None))

    assert np.all(dist_analy.build_shortest_dist_matrix(res_obj, res_list, min_dist = 2.2).diagonal() == 0 )

@pytest.mark.parametrize('res_list', [CDK2_KLIFS_IDENT])
def test_compare_to_chimera(res_obj, res_list):
    """Comparing dist_analy package results to chimera results

    Parameters
    ----------
    res_obj : np.array
        residue object pytest fixture
    res_list : list
        residue list

    """
    KI_sd_npy_file = './datafiles/npy_files/ref/4EOQ.pdb_KI_sd.npy'
    KI_sd_noadj_npy_file = './datafiles/npy_files/ref/4EOQ.pdb_KI_sd_noadj.npy'

    KI_sd = np.around(np.load(KI_sd_npy_file),5)
    KI_sd_noadj = np.around(np.load(KI_sd_noadj_npy_file), 5)
    #
    # print(dist_analy.build_shortest_dist_matrix(res_obj, res_list))

    dist_mat_sd = np.around(dist_analy.build_shortest_dist_matrix(res_obj, res_list, no_adj = False),5)
    dist_mat_sd_noadj = np.around(dist_analy.build_shortest_dist_matrix(res_obj, res_list, no_adj = True),5)


    assert len(dist_mat_sd ) == len(KI_sd)
    assert np.array_equal(dist_mat_sd, KI_sd)
    #
    # for i in range(len(KI_sd)):
    #     if not np.array_equal(dist_mat_sd_noadj[i], KI_sd_noadj[i]):
    #         print (dist_mat_sd_noadj[i], KI_sd_noadj[i])
    #         truthy = np.equal(dist_mat_sd_noadj[i], KI_sd_noadj[i])
    #         for z,x in enumerate(truthy):
    #             if x == False:
    #                 print(res_list[z], dist_mat_sd_noadj[i][z], KI_sd_noadj[i][z])
    #         print (i, [j for j,x in enumerate(truthy) if x == False], truthy)

    assert len(dist_mat_sd_noadj) == len(KI_sd_noadj)
    assert np.array_equal(dist_mat_sd_noadj, KI_sd_noadj)

# MISSING = [40,41,42] # testing for glycine and no_adj
@pytest.mark.parametrize('res_list', [MISSING])
def test_missing_residues(res_obj, res_list):
    """Testing how the program handles missing residues within the 4EOQ structure

    Parameters
    ----------
    res_obj : np.array
        residue object pytest fixture
    res_list : list
        residue list

    """
    mat = dist_analy.build_shortest_dist_matrix(res_obj, res_list, no_adj = True)
    assert len(mat[0]) == len(MISSING)
    mat = dist_analy.build_shortest_dist_matrix(res_obj, res_list, no_adj = True, min_dist = 2.2)
    assert len(mat[0]) == len(MISSING)


@pytest.mark.parametrize('res_list', [CDK2_KLIFS_IDENT])
def test_get_shortest_dist_matrix(res_list):
    """Tests get_shortest_dist_matrix
    """
    file = './datafiles/pdb_files/testing/4EOQ.pdb'
    chain = 'A'
    dist_analy.get_shortest_dist_matrix(file, res_list, chain)
    dist_analy.get_shortest_dist_matrix(file, res_list, chain, min_dist = 2.2)
    dist_analy.get_shortest_dist_matrix(file, res_list, chain, min_dist = 2.2, no_adj = True)
    #
    # structure= pdbfile.parsePDB(file, chain=chain)
    # hv = structure.getHierView()
    # res_list_2 = [4,5]
    # res_obj_1= np.array(hv[chain], dtype=Residue)[res_list]
    # res_obj_2= np.array(hv[chain], dtype=Residue)[res_list_2]
    # dist_analy.build_shortest_dist_matrix(res_obj_1, res_list, res_obj_2, res_list_2, format = 'rcd')
