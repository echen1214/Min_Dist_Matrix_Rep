# import os
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO, AlignIO
from Bio.Align.Applications import ClustalwCommandline
from dist_analy.util import pdb_info

"""
TODO
- implement flexible reading of different sequence alignment tools
    - Bio is flexible so should maybe allow different methods and different
    alignments -> test all combinations of getting sequence and aligning
- automatically find and search for the sequence alignment tool

"""
uniprot_seq_url = 'http://www.ebi.ac.uk/proteins/api/proteins/'
clustalw_exe = '/Users/echen10/Desktop/programs/clustal-omega-1.2.3-macosx'


def get_uniprot_sequence(uniprot: str, prot: str, description: str = ''):
    """ helper function for getting the uniprot sequence

    Parameters
    ----------
    uniprot : str
        uniprot code
    prot : str
        name of the protein to store
    description : str
        any additional description

    Returns
    -------
    SeqRecord object
        Biopython sequence object

    """
    uniprot_info = pdb_info.get_any_info('http://www.ebi.ac.uk/proteins/api/proteins/', uniprot)
    temp_rec = SeqRecord(Seq(uniprot_info['sequence']['sequence']), id=uniprot, name=prot, description = description)

    if not temp_rec:
        raise ValueError("unable to get uniprot sequence for %s"%prot)
    return temp_rec

def get_write_fasta(outpath: str, outfile: str, unip_list: list, prot_list: list, \
             descrip_list: list):
    """ Retrieving and writing out a series of uniprot IDs and their sequences
    to the specified file. Each index of the uniprot list, protein name list,
    and description list should correspond with one another

    todo flexible choice of output

    Parameters
    ----------
    outpath : str
        path to write the file
    outfile : str
        name of the file
    unip_list : list
        list of uniprot IDs
    prot_list : list
        list of protein names
    descrip_list : list
        list of descriptions

    """
    Path(outpath).mkdir(parents=True, exist_ok=True)
    seq_list = []
    for unip, prot, desc in zip(unip_list, prot_list, descrip_list):
        seq_list.append(get_uniprot_sequence(unip, prot, desc))

    with open(outpath+outfile, "w") as outf:
        SeqIO.write(seq_list, outf, "fasta")

def align_fasta_clustalw(inpath: str, infile: str, outpath: str, outfile: str):
    """Utilizes the Biopython align applications package to run a command line
    alightment program and write out an alignment

    todo: add a methods option for flexible choice of alignment program

    Parameters
    ----------
    inpath : str
        path of the fasta file
    infile : str
        name of the file
    outpath : str
        path of the output file
    outfile : str
        name of the outfile

    """
    Path(outpath).mkdir(parents=True, exist_ok=True)
    clustalw_cline = ClustalwCommandline(clustalw_exe, infile=inpath+infile, outfile=outpath+outfile)
    stdout, stderr = clustalw_cline()

def get_and_align_sequence(filename: str, outpath: str, unip_list: list, prot_list: list = [], \
                           descrip_list: list = []):
    """Function that creates a file of the sequences and then runs a command line
    sequence alignment. Then it loads the alignment into the Biopython align
    object

    Parameters
    ----------
    filename : str
        name of the root file name for the sequence file and alignment
    outpath : str
        path of produce the output to
    unip_list : list
        list of uniprot IDs
    prot_list : list
        list of protein names
    descrip_list : list
        list of descriptions

    Returns
    -------
    Align object
        Biopython align object

    """
    Path(outpath).mkdir(parents=True, exist_ok=True)
    if not prot_list:
        prot_list = ['' for _ in range(len(unip_list))]
    if not descrip_list:
        descrip_list = ['' for _ in range(len(unip_list))]

    f1, f2 = filename, filename
    if len(filename.split('.')) == 1:
        f1+='.fa'
        f2+='.al'
    get_write_fasta(outpath+'fasta/', f1,  unip_list, prot_list, descrip_list)
    align_fasta_clustalw(outpath+'fasta/', f1, outpath+'align/', filename+'.al')

    align = AlignIO.read(outpath+'align/'+f2, "clustal")
    return align

def get_conserved(align):
    """Determines the residue IDs of the identically conserved residues of an
    sequence alignment for each corresponding sequence

    Parameters
    ----------
    align : Align biopython object
        sequence alignment

    Returns
    -------
    list
        list of identically conserved residue lists. Each list index corresponds
        the sequence in the alignment

    """
    align_seq = [obj.seq for obj in align]
    res_count = [0 for obj in align]
    cons_id = [[] for obj in align]

    for i in range(0,len(align_seq[0])):
        temp_set = set()
        for j,seq in enumerate(align_seq):
            temp_set.add(seq[i])
            if seq[i]!='-':
                res_count[j] += 1
        if len(temp_set) == 1:
            for j,res in enumerate(res_count):
                cons_id[j].append(res)
    return cons_id

def get_klifs_res(ref_pdb: str, chain: str):
    """ Makes request to the KLIFs api for the Xray positions of the KLIFs residues

    Parameters
    ----------
    ref_pdb : str
        pdb id
    chain : str
        chain

    Returns
    -------
    list
        list of KLIFs residues

    """
    pdb = ref_pdb.split('.')[0]
    request_dict = {
        "pdb-codes": [pdb]
    }

    struct_list = pdb_info.get_any_info('https://klifs.net/api/structures_pdb_list', **request_dict)

    structure_id = 0
    for struct in struct_list:
        # print(struct['pdb'], pdb.lower(), struct['chain'], chain)
        if struct['pdb'] == pdb.lower() and struct['chain'] == chain:
            structure_id = struct['structure_ID']

    if not structure_id:
        raise ValueError("structure ID was not found for pdb %s chain %s"%(pdb,chain))

    request_dict = {
        "structure_ID": structure_id
    }

    klifs_res_list = pdb_info.get_any_info('https://klifs.net/api/interactions_match_residues', **request_dict)

    klifs_res = [int(temp_dict['Xray_position']) for temp_dict in klifs_res_list if temp_dict['Xray_position'] != '_']
    return klifs_res

def def_union(s1,s2):
    """Given two lists of numbers return the union of the two list

    Parameters
    ----------
    s1 : list
        first list
    s2 : list
        second list

    Returns
    -------
    list, list
        sorted list of numbers representing the union of the two sets
        list of numbers reflecting the corresponding index of the union and
        whether the number came from the list 1 (1), list 2 (2) or both (0)

    """
    # returns two arrays 1) union of two sets and 2) array where each index reflects which set
    # the index came from. 0 from both, 1 from s1, 2 from s2
    union=sorted(set(s1).union(set(s2)))
    union_ind = []
    for x in union:
        if x in s1 and x in s2:
            union_ind.append(0)
        elif x in s1:
            union_ind.append(1)
        elif x in s2:
            union_ind.append(2)
    return(union, union_ind)
