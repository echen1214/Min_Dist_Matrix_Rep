import numpy as np
import rdkit.Chem as Chem

from spyrmsd import io, rmsd
from scipy.spatial import distance
from Bio.PDB import Superimposer, PDBParser
from Bio.PDB.PDBIO import Select, PDBIO
from Bio.PDB.PDBExceptions import PDBIOException

from meeko import PDBQTMolecule, RDKitMolCreate
from spyrmsd.molecule import Molecule

metric_list = {
    "all_RMSD_lt2": ("RMSD", 2, 10, "confidence"),
    "all_RMSD_lt5": ("RMSD", 5, 10, "confidence"),
    "top5_RMSD_lt2": ("RMSD", 2, 5, "confidence"),
    "top5_RMSD_lt5": ("RMSD", 5, 5, "confidence"),
    "all_cent_dist_lt2": ("cent_dist", 2, 10, "confidence"),
    "all_cent_dist_lt5": ("cent_dist", 5, 10, "confidence"),
    "top5_cent_dist_lt2": ("cent_dist", 2, 5, "confidence"),
    "top5_cent_dist_lt5": ("cent_dist", 5, 5, "confidence"),
}

metric_list_vina = {
    "all_RMSD_lt2": ("RMSD", 2, 10, "pVina_LRD"),
    "all_RMSD_lt5": ("RMSD", 5, 10, "pVina_LRD"),
    "top5_RMSD_lt2": ("RMSD", 2, 5, "pVina_LRD"),
    "top5_RMSD_lt5": ("RMSD", 5, 5, "pVina_LRD"),
    "all_cent_dist_lt2": ("cent_dist", 2, 10, "pVina_LRD"),
    "all_cent_dist_lt5": ("cent_dist", 5, 10, "pVina_LRD"),
    "top5_cent_dist_lt2": ("cent_dist", 2, 5, "pVina_LRD"),
    "top5_cent_dist_lt5": ("cent_dist", 5, 5, "pVina_LRD"),
}

metric_list_linf9 = {
    "all_RMSD_lt2": ("RMSD", 2, 10, "pLin_F9_LRD"),
    "all_RMSD_lt5": ("RMSD", 5, 10, "pLin_F9_LRD"),
    "top5_RMSD_lt2": ("RMSD", 2, 5, "pLin_F9_LRD"),
    "top5_RMSD_lt5": ("RMSD", 5, 5, "pLin_F9_LRD"),
    "all_cent_dist_lt2": ("cent_dist", 2, 10, "pLin_F9_LRD"),
    "all_cent_dist_lt5": ("cent_dist", 5, 10, "pLin_F9_LRD"),
    "top5_cent_dist_lt2": ("cent_dist", 2, 5, "pLin_F9_LRD"),
    "top5_cent_dist_lt5": ("cent_dist", 5, 5, "pLin_F9_LRD"),
}

metric_list_dyn = {
    "all_RMSD_lt2": ("RMSD", 2, 10, "affinity"),
    "all_RMSD_lt5": ("RMSD", 5, 10, "affinity"),
    "top5_RMSD_lt2": ("RMSD", 2, 5, "affinity"),
    "top5_RMSD_lt5": ("RMSD", 5, 5, "affinity"),
    "all_cent_dist_lt2": ("centroid_distance", 2, 10, "affinity"),
    "all_cent_dist_lt5": ("centroid_distance", 5, 10, "affinity"),
    "top5_cent_dist_lt2": ("centroid_distance", 2, 5, "affinity"),
    "top5_cent_dist_lt5": ("centroid_distance", 5, 5, "affinity"),
}


# https://spyrmsd.readthedocs.io/en/develop/
def load_fn(fn):
    return io.loadmol(fn)
def load_pdbqt(fn):
    with open(fn) as f:
        ## load pdbqt file into meeko and convert to rdkit molecule
        string = f.read()
        pdbqt_mol = PDBQTMolecule(string, is_dlg=False, poses_to_read=1, skip_typing=True)
        rdmol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        ## convert into spyrmsd molecule
        spymol = Molecule.from_rdkit(rdmol[0])
    return(spymol)
def load_out_str(out_str):
    rdmol = Chem.MolFromPDBBlock("".join(out_str))
    mol = io.to_molecule(rdmol)
    return(mol)
def load(mol, mol_type):
    if mol_type=="fn":
        return load_fn(mol) 
    elif mol_type=="pdbqt":
        return load_pdbqt(mol) 
    elif mol_type=="out_str":
        return load_out_str(mol)

def docking_accuracy(df_per_recep, metric="RMSD", lt=2, top=20, sortby="confidence", res_num=False):
    if res_num:
        # multiple reference ligands that we calculated RMSD to...just select the lowest RMSD
        val = np.min([r[metric].values for k, r in df_per_recep.sort_values(sortby, ascending=False).groupby("res num", sort=False).head(top).groupby("res num", sort=False)], axis=0)
    else:
        val = df_per_recep.sort_values(sortby, ascending=False).head(top)[metric].values
    val_out = 1 if any(val < lt) else 0
    return(val_out)

def sRMSD(mol1, mol2, no_H=True, mol1_type="fn",  mol2_type="fn"):
    assert (mol1_type in ["fn", "pdbqt", "out_str"]) and (mol2_type in ["fn", "pdbqt", "out_str"])
    ref = load(mol1, mol1_type)
    mol = load(mol2, mol2_type)

    if ref:
        anum_ref = ref.atomicnums
        coords_ref = ref.coordinates
        adj_ref = ref.adjacency_matrix
        if no_H:
            noH_ind = np.where(anum_ref!=1)[0]
            coords_ref = coords_ref[noH_ind]
            adj_ref = adj_ref[np.ix_(noH_ind, noH_ind)]
            anum_ref = anum_ref[noH_ind]
    else:
        raise ValueError(f"Unable to load {ref_fn}") 

    if mol:
        coords = mol.coordinates
        anum = mol.atomicnums
        adj = mol.adjacency_matrix
        if no_H:
            noH_ind = np.where(anum!=1)[0]
            coords = coords[noH_ind]
            adj = adj[np.ix_(noH_ind, noH_ind)]
            anum = anum[noH_ind]
    else:
        raise ValueError(f"Unable to load {mol_fn}") 
    RMSD = rmsd.symmrmsd(coords_ref,coords,anum_ref,anum,adj_ref,adj)
    cent_dist = distance.euclidean(np.mean(coords_ref,axis=0), np.mean(coords,axis=0))

    return(RMSD, cent_dist)

def pad_square_symmat_with_vector(mat, vector):
    padded_vector = np.append(vector, 0)
    new_mat = np.zeros((len(padded_vector), len(padded_vector)))
#     print(new_mat.shape, padded_vector.shape, padded_vector)
    new_mat[:-1, :-1] = mat
    new_mat[-1, :] = padded_vector
    new_mat[:, -1] = padded_vector
    return(new_mat)


# https://spyrmsd.readthedocs.io/en/develop/
def get_crd_symmetry_rmsd(mol_str, pred_mol, pred_coords, sub_orig_center_from_ref = None, no_H=True):

    mol = molecule.Molecule.from_rdkit(pred_mol)
    
    rdmol = Chem.MolFromPDBBlock("".join(mol_str))
    ref = io.to_molecule(rdmol)

    if ref:
        anum_ref = ref.atomicnums
        coords_ref = ref.coordinates
        adj_ref = ref.adjacency_matrix
        if no_H:
            noH_ind = np.where(anum_ref!=1)[0]
            coords_ref = coords_ref[noH_ind]
            adj_ref = adj_ref[np.ix_(noH_ind, noH_ind)]
            anum_ref = anum_ref[noH_ind]
        if not isinstance(sub_orig_center_from_ref, type(None)):
            coords_ref -= sub_orig_center_from_ref
    else:
        raise ValueError(f"Unable to load {ref_fn}") 

    if mol:
        coords = pred_coords
        anum = mol.atomicnums
        adj = mol.adjacency_matrix
        if no_H:
            noH_ind = np.where(anum!=1)[0]
            coords = [c[noH_ind] for c in coords]
            adj = adj[np.ix_(noH_ind, noH_ind)]
            anum = anum[noH_ind]
    else:
        raise ValueError(f"Unable to load {mol_fn}") 

    RMSD = rmsd.symmrmsd(coords_ref,coords,anum_ref,anum,adj_ref,adj)
    cent_dist = np.linalg.norm(np.mean(coords,axis=1)-np.mean(coords_ref,axis=0), axis=1)

    return(RMSD, cent_dist)

def sRMSD_from_str(ref_fn, mol_str, no_H=True):
    ref = io.loadmol(ref_fn)
    rdmol = Chem.MolFromPDBBlock("".join(mol_str))
    mol = io.to_molecule(rdmol)

    if ref:
        anum_ref = ref.atomicnums
        coords_ref = ref.coordinates
        adj_ref = ref.adjacency_matrix
        if no_H:
            noH_ind = np.where(anum_ref!=1)[0]
            coords_ref = coords_ref[noH_ind]
            adj_ref = adj_ref[np.ix_(noH_ind, noH_ind)]
            anum_ref = anum_ref[noH_ind]
    else:
        raise ValueError(f"Unable to load {ref_fn}") 

    if mol:
        coords = mol.coordinates
        anum = mol.atomicnums
        adj = mol.adjacency_matrix
        if no_H:
            noH_ind = np.where(anum!=1)[0]
            coords = coords[noH_ind]
            adj = adj[np.ix_(noH_ind, noH_ind)]
            anum = anum[noH_ind]
    else:
        raise ValueError(f"Unable to load {mol_fn}") 
    RMSD = rmsd.symmrmsd(coords_ref,coords,anum_ref,anum,adj_ref,adj)
    cent_dist = distance.euclidean(np.mean(coords_ref,axis=0), np.mean(coords,axis=0))

    return(RMSD, cent_dist)


def save_str(io: PDBIO, chain, write_end=True, preserve_atom_numbering=True, select = Select()):
    # https://github.com/biopython/biopython/blob/master/Bio/PDB/PDBIO.py#L297C42-L297C87
    # using Biopython, pass in a PBDIO object, select a chain and any other specifics
    # from the Select() object, and output the results as a string in PDB format
    out = []
    get_atom_line = io._get_atom_line
    chain = io.structure[0][chain]
    chain_id = chain.id
    if len(chain_id) > 1:
        e = f"Chain id ('{chain_id}') exceeds PDB format limit."
        raise PDBIOException(e)

    chain_residues_written = 0

    for residue in chain.get_unpacked_list():
        if not select.accept_residue(residue):
            continue
        hetfield, resseq, icode = residue.id
        resname = residue.resname
        segid = residue.segid
        resid = residue.id[1]
        if resid > 9999:
            e = f"Residue number ('{resid}') exceeds PDB format limit."
            raise PDBIOException(e)

        for atom in residue.get_unpacked_list():
            if not select.accept_atom(atom):
                continue
            chain_residues_written = 1
            model_residues_written = 1
            if preserve_atom_numbering:
                atom_number = atom.serial_number

            try:
                s = get_atom_line(
                    atom,
                    hetfield,
                    segid,
                    atom_number,
                    resname,
                    resseq,
                    icode,
                    chain_id,
                )
            except Exception as err:
                # catch and re-raise with more information
                raise Exception(
                    f"Error when writing atom {atom.full_id}"
                ) from err
            else:
                out.append(s)
                # inconsequential if preserve_atom_numbering is True
                atom_number += 1
    return out

def load_struct(fn_list):
    pdb_parser = PDBParser(QUIET = True)
    struct_dict = {}
    for tup in fn_list:
        fn, chain, pdb = tup
        struct_dict[pdb] = {
            'chain': chain,
            'struct': pdb_parser.get_structure(pdb, fn)[0][chain]
        }
    return(struct_dict)

def align_lig(ref_struct, samp_struct, samp_lig, lig_chain, res_list):
    ref_atoms_align, samp_atoms_align = [], []

    for res in res_list:
        try:
            ref = ref_struct.__getitem__(res).__getitem__("CA")
            samp = samp_struct.__getitem__(res).__getitem__("CA")
            ref_atoms_align.append(ref)
            samp_atoms_align.append(samp)
        except KeyError:
            pass
    
    supimp = Superimposer()
    pdbio = PDBIO()

    supimp.set_atoms(ref_atoms_align, samp_atoms_align)
    supimp.apply(samp_lig.get_atoms())
    pdbio.set_structure(samp_lig)

    out_str = save_str(pdbio, lig_chain) # select = SelLig_RejectWater(lig, res_num))
    return(out_str)

def get_sd_results(df, metric_list):
    metric_self_docking = {}
    for key, metric in metric_list.items():
        tmp = {}
        for ro_lo, df_tmp in df.groupby(["ro_label", "lo_label"]):
            per_pdb_chain = [docking_accuracy(per_rec, *metric,) for m, per_rec in df_tmp.groupby("LIG")]
            # print(ro_lo, per_pdb_chain, f"{sum(per_pdb_chain)/len(per_pdb_chain):0.3f}")
            tmp[ro_lo] = {"per_pdb_chain": per_pdb_chain, "average":f"{sum(per_pdb_chain)/len(per_pdb_chain):0.3f}" }
        metric_self_docking[key] = tmp
    return metric_self_docking

def get_crd_results(df, metric_list):
    metric_per_rec_crossdocking = {}

    for key, metric in metric_list.items():
        per_rec_crossdocking = {x:{} for x in df.lig_lr_label.unique()}
        for lr_ro, df_tmp in df.groupby(["lig_lr_label", "ro_label"]):
            per_pdb_chain = [docking_accuracy(per_rec, *metric,) for m, per_rec in df_tmp.groupby("PDB_chain_LIG")]
            pdb_lig = [per_rec.PDB_chain_LIG.values[0] for m, per_rec in df_tmp.groupby("LIG")]
            per_rec_crossdocking[lr_ro[0]][lr_ro[1]] = f"{sum(per_pdb_chain)/len(per_pdb_chain):.3f}"
        metric_per_rec_crossdocking[key]=per_rec_crossdocking

    return metric_per_rec_crossdocking
