import os
import sys
import time

from prody import *
from dist_analy import *


# this function was used to perform actual analysis for my research
def analyze_residues():
    create_chimera_script(
        "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test",
        [
            "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/5CAU_A.pdb",
            "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/6V5P_D.pdb",
        ],
        [
            [745, 762],
        ],
    )


# this function was created for testing functionality of the main function
def test_function():
    """
    file_split_1 = "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/"
    file_split_2 = "1IVO_A.pdb"

    get_shortest_atom_distance(file_split_1, file_split_2)
    """

    create_chimera_script(
        "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test",
        [
            "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/4ZSE_D.pdb",
            "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/3LZB_C.pdb",
            "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/5HIC_A.pdb",
            "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/6V6O_H.pdb",
            "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/8D73_A.pdb",
        ],
        [[847, 848], [855, 842]],
    )

    """ get_shortest_atom_distance_type(
        "/Users/jsarnell/Desktop/REUProject/dist_analy/john_test/test_pdb/4ZSE_D.pdb",
        "D",
        [[847, 848], [855, 842]]
    ) """


def get_shortest_atom_distance_type(
    pdb: str,
    chain: str,
    residue_list: list,
    include_backbone: bool = True,
    include_heavy_residues: bool = True,
):
    """Takes in the necessary parameters for a specific PDB and outputs the type
        of the atoms which are the closest together in a specified residue pair

    Parameters
    ----------
    pdb : str
        The file path to the pdb being input
    chain : str
        The specific chain in the PDB you would like analyzed
    residue_list : list
        A list of the residue pairs in the molecule you want analyzed
    include_backbone : bool, optional
        A flag that allows you to include or exclude the backbone atoms of a residue
        Automatically overriden to False when the residues are adjacent to each other
    include_heavy_residues: bool, optional
        When set to true, only includes heavy atoms in the residue, false includes all atoms
    """
    # Set up variables for function
    min_distance = sys.maxsize
    residues = get_res_obj(pdb, chain, residue_list)

    # Does not include backbone if residues are adjacent
    if (residues[0].getResnum() - residues[1].getResnum()) == 1 or (
        residues[0].getResnum() - residues[1].getResnum()
    ) == -1:
        include_backbone = False

    residue_coords = get_atom_coords(
        include_heavy_residues, residues, Residue, include_backbone
    )

    # Calculations to find atoms that are closest together in specified residues
    for count_1, coords_1 in enumerate(residue_coords[0]):
        for count_2, coords_2 in enumerate(residue_coords[1]):
            if prody.calcDistance(coords_1, coords_2) < min_distance:
                atom_1_type = residues[0].getNames()[count_1]
                atom_2_type = residues[1].getNames()[count_2]
                min_distance = prody.calcDistance(coords_1, coords_2)

    print(min_distance)
    # return the type of the two closest atoms
    return atom_1_type, atom_2_type


def write_chimera_script(count: int, fhand, pdb: str, residue_list: list):
    """Gets the necessary data to write commands to a Chimera script and then
        writes commands

    Parameters
    ----------
    count : int
        Keeps track of the number of PDBs, specifically for use in aligning structures
    fhand
        The file object associated with the file that is being written to
    pdb : str
        The file path to the pdb being input
    residue_list : list
        A list of the residue pairs in the molecule you want analyzed
    """

    # write command to open pdb
    fhand.write(f"open {pdb}\n")

    # prepare parameters for get_shortest_atom_distance_type
    chain = pdb.split(".")[0].split("_")[-1]

    # write commands to draw distances between all residue pairs
    for residue_pair in residue_list:
        (atom_1, atom_2) = get_shortest_atom_distance_type(pdb, chain, residue_pair)
        fhand.write(
            f"distance #{count}/{chain}:{residue_pair[0]}@{atom_1} #{count}/{chain}:{residue_pair[1]}@{atom_2}\n"
        )

    # write commands to align structures
    if count != 1:
        fhand.write(f"mm #{str(count)} to #1 showAlignment true\n")


def create_chimera_script(
    file_path_script: str,
    pdb_list: list,
    residue_list: list,
    file_name: str = "default_chimera_script.cxc",
):
    """Creates a Chimera script to display and calculate distances between atoms
    from specificed PDBs in Chimera

    Parameters
    ----------
    file_path_script : str
    |   The filepath to the directory where you want to store the script
    pdb_list : str
    |   The filepath to the directory containing the PDBs
    residue_list : list
        A list of the residue pairs in the molecule you want analyzed
    file_name : str
    |   The name you want to set for the script
    """

    # open the chimera script file and then iterate through the PDBs, calling
    #    write_chimera_script as necessary
    with open(os.path.join(file_path_script, file_name), "w") as fhand:
        count = 1
        for pdb in pdb_list:
            write_chimera_script(count, fhand, pdb, residue_list)
            if count == 1:
                count += 2
            else:
                count += 1
        # final quality of life commands
        # -> makes anaylsis of distances easier
        for residue_pair in residue_list:
            # all atoms are shown for clearer analysis in chimera -> might want to try to move this command somewhere else. Currently runs more times than it needs to
            fhand.write(
                f"show :{residue_pair[0]} :{residue_pair[1]}\n~rib :{residue_pair[0]} :{residue_pair[1]}\n"
            )
        fhand.write("label height 1.5\n")
        fhand.write("distance style radius 0.2\n")
        fhand.write("transparency 80 ribbons\n")


# This is the updated version of get_atom_coords
# I am fairly certain that if you don't use this version, my code will throw errors
def get_atom_coords(heavy: bool, residues1: List, type, backbone: bool = True):
    if heavy and backbone:
        atomcoords1 = np.array(
            [
                x.select("heavy").getCoords() if isinstance(x, type) else None
                for x in residues1
            ],
            dtype=ndarray,
        )
    elif heavy:
        atomcoords1 = np.array(
            [
                x.select("heavy and not backbone").getCoords()
                if isinstance(x, type)
                else None
                for x in residues1
            ],
            dtype=ndarray,
        )
    elif not backbone:
        atomcoords1 = np.array(
            [
                x.select("not backbone").getCoords() if isinstance(x, type) else None
                for x in residues1
            ],
            dtype=ndarray,
        )
    else:
        atomcoords1 = np.array(
            [x.getCoords() if isinstance(x, type) else None for x in residues1],
            dtype=ndarray,
        )
    return atomcoords1


def main():
    # test_function()
    analyze_residues()


if __name__ == "__main__":
    main()
