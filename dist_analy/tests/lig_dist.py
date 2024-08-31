#!/usr/bin/env python

import numpy as np
from dist_analy import dist_analy


rec_pdb = './datafiles/pdb_files/processed_pdb/1P2A_A_01.pdb'
res_list = np.arange(1,299)
# save_fn = "./datafiles/npy_files/ref_lig/dd_lrd/1P2A_A_01_rank1-20_UNL1.npy"
# lig_list = [f"./datafiles/pdb_files/ligand_poses/diffdock_pose/rank{i}.pdb" for i in range(1,21)]
save_fn = "./datafiles/npy_files/ref_lig/pdb_pose/1P2A_A_5BN_301.npy"
lig_fn = "./datafiles/pdb_files/ligand_poses/pdb_pose/1P2A_A_5BN_301.pdb"
mat = dist_analy.get_shortest_dist_matrix(rec_pdb, res_list, "A", ligand_file=lig_fn, save_fn=save_fn)
print(mat.shape)
print(np.load(save_fn).shape)