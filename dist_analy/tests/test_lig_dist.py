#!/usr/bin/env python

"""Tests for `dist_analy` package.
- test for multiple chains (reading 1 pdb file with 2 chains of the uniprot protein
    should result in two different structures and distance matrices)
- test for mismatching sequence numbering
- test for multiple occupancies/models
"""

import pytest
import numpy as np
from dist_analy import dist_analy

pdb_path = "./datafiles/pdb_files/ligand_poses/"
npy_path = "./datafiles/npy_files/ref_lig/"

def test_no_chain_res_obj():
    rec_pdb = './datafiles/pdb_files/processed_pdb/1P2A_A_01.pdb'
    res_list = np.arange(1,299)
    
    no_chain = "./datafiles/pdb_files/ligand_poses/diffdock_pose/rank1.pdb"
    chain = "./datafiles/pdb_files/ligand_poses/diffdock_pose/rank1_chainA.pdb"
    # assert dist_analy.get_res_obj(no_chain) == dist_analy.get_res_obj(chain, chain="A")
    mat = dist_analy.get_shortest_dist_matrix(rec_pdb, res_list, "A", ligand_file=no_chain)
    comp = dist_analy.get_shortest_dist_matrix(rec_pdb, res_list, "A", ligand_file=no_chain, ligand_chain="A")
    assert np.allclose(mat, comp)

@pytest.mark.parametrize(
    'pdb_file, npy_file', 
    [(f"{pdb_path}/diffdock_pose/rank{i}.pdb", 
    f"{npy_path}dd_lrd/1P2A_A_01_rank{i}_UNL1.npy") for i in range(1,5)] +  
    [(f"{pdb_path}pdb_pose/1P2A_A_5BN_301.pdb", f"{npy_path}pdb_pose/1P2A_A_5BN_301.npy")]
)
def test_dd_lrd_pose_and_compare(pdb_file, npy_file):
    rec_pdb = './datafiles/pdb_files/processed_pdb/1P2A_A_01.pdb'
    res_list = np.arange(1,299)
    mat = dist_analy.get_shortest_dist_matrix(rec_pdb, res_list, "A", ligand_file=pdb_file)
    comp = np.load(npy_file)

    assert np.allclose(mat, comp)
    
@pytest.mark.parametrize(
    'pdb_file_list, npy_file', 
    [([f"./datafiles/pdb_files/ligand_poses/diffdock_pose/rank{i}.pdb" for i in range(1,21)], 
    f"./datafiles/npy_files/ref_lig/dd_lrd/1P2A_A_01_rank1-20_UNL1.npy")]
)
def test_dd_lrd_pose_bulk_and_compare(pdb_file_list, npy_file):
    rec_pdb = './datafiles/pdb_files/processed_pdb/1P2A_A_01.pdb'
    res_list = np.arange(1,299)
    mat = dist_analy.get_shortest_dist_matrix(rec_pdb, res_list, "A", ligand_file=pdb_file_list)
    comp = np.load(npy_file)
    assert mat.shape == comp.shape
    assert np.allclose(mat, comp)
