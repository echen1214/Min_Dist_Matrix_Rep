from io import BytesIO
from ftplib import FTP
import gzip
from numpy import all
import pytest
from Bio.PDB import PDBParser, PDBIO
from dist_analy import import_pdb
from dist_analy.util import pdb_info

""" List of things to test
- test for multiple chains (reading 1 pdb file with 2 chains of the uniprot protein
    should result in two different structures and distance matrices)
- test for mismatching sequence numbering
- test for multiple occupancies/models
"""

PDB_DIR = './datafiles/pdb_files/testing/'
OUTPATH = './datafiles/pdb_files/processed_pdb/'
CHECKPATH = './datafiles/pdb_files/check_pdb/'
TEST_PDB_LIST = ['2VU3.pdb', '5OO0.pdb']
TEST_CHAIN = ['A', 'A']
MISNUMBERED_PDB = ['4JRV.pdb', '2ABL.pdb' ] # egfr1/ abl
# MULTIPLE_OCCU_PDB = ['2R3J.pdb', '2R3I.pdb', '2R3Q.pdb']
# SET_MULTIPLE_OCCU = [(32, 115, 177, 200, 212, 232, 233), \
#         (32, 53, 115, 126, 131, 177, 200, 212, 217, 232, 233, 264, 265, 501), \
#         (89, 126, 131, 177, 200, 212, 217, 232, 233, 264, 265, 501)]
MULTIPLE_CHAIN_PDB = ['2WPA.pdb','4EOQ.pdb', '6GUE.pdb']
MULTIPLE_CHAIN = ['A,C', 'A,C', 'A']
UNIPROT_SEQ = ["MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNH\
    PNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHS\
    HRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYY\
    STAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSF\
    PKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"]
UNIPROT_CDK2 = ['P24941']
UNIPROT_EGFR1 = ['P00533']
UNIPROT_ABL = ['P00519']

# @pytest.mark.parametrize("pdb_fn, chain",zip(TEST_PDB_LIST+MULTIPLE_CHAIN_PDB, TEST_CHAIN+MULTIPLE_CHAIN))
# def test_get_any_info_1(pdb_fn, chain):
#     pdb = pdb_fn.split('.')[0]
#     uniprot_url = 'https://data.rcsb.org/rest/v1/core/uniprot/'
#     entity_url = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'
#     assert pdb_info.get_any_info(uniprot_url, pdb, str(1))[0]['rcsb_uniprot_container_identifiers']['uniprot_id'] == 'P24941'
#     assert pdb_info.get_any_info(entity_url, pdb, str(1))['entity_poly']['pdbx_strand_id'] == chain

def chain_obj(pdb_fn, chain):
    pdb = pdb_fn.split('.')[0]
    parse = PDBParser()
    structure = parse.get_structure(pdb, file=pdb_fn)[0]
    return structure[chain]
#
# @pytest.mark.parametrize("pdb_fn, chain", zip(MISNUMBERED_PDB, 'A'))
# def test_shift_res(pdb_fn, chain):
#     struct1 = chain_obj(pdb_fn, chain)
#     shift_res(struct1, 24)
#     pdb = pdb_fn.split('.')[0]
#     comp = chain_obj(CHECKPATH+pdb+'_'+chain+'.pdb', chain)
#     assert struct1 == comp

@pytest.fixture
def xml_str(pdb_fn):
    pdb = pdb_fn.split('.')[0].lower()
    ftp_url = 'ftp.ebi.ac.uk' #pub/databases/msd/sifts/xml/%s.xml.gz'%pdb
    ftp = FTP(ftp_url)
    ftp.login()

    BIO = BytesIO()
    # print("RETR /pub/databases/msd/sifts/xml/%s.xml.gz"%pdb)
    ftp.retrbinary("RETR /pub/databases/msd/sifts/xml/%s.xml.gz"%pdb, callback=BIO.write)
    BIO.seek(0) # Go back to the start
    zippy = gzip.GzipFile(fileobj=BIO)

    uncompressed = zippy.read()
    ftp.quit()
    return uncompressed

@pytest.mark.parametrize("pdb_fn, answer", zip(MISNUMBERED_PDB+TEST_PDB_LIST, [False,False,True,True]))
def test_xml_replace_dict(xml_str, answer):
    pdb_proc = import_pdb.PDB_Processer()
    repl_dict = pdb_proc._xml_replace_dict(xml_str, 'A')
    truthy = []
    print(repl_dict)
    for key in repl_dict.keys():
        truthy.append(key == repl_dict[key])
    assert all(truthy) == answer

@pytest.mark.parametrize("pdb_fn, uniprot",zip(MISNUMBERED_PDB, UNIPROT_EGFR1+UNIPROT_ABL))
def test_process_pdb_1(pdb_fn, uniprot):
    pdb_proc = import_pdb.PDB_Processer(['TPO'])
    chain = 'A'
    pdb = pdb_fn.split('.')[0]
    pdb_proc.process_pdb(pdb_fn, PDB_DIR, OUTPATH, uniprot)

    comp = chain_obj(OUTPATH+pdb+'_'+chain+'.pdb', chain)
    ref = chain_obj(CHECKPATH+pdb+'_'+chain+'.pdb', chain)

    res_list_comp = [list(res.id)[1] for res in comp.get_residues()]
    res_list_ref = [list(res.id)[1] for res in ref.get_residues()]

    assert all(res_list_comp == res_list_ref)

@pytest.mark.parametrize("pdb_fn, uniprot",zip (MULTIPLE_CHAIN_PDB, UNIPROT_CDK2*3))
def test_process_pdb_2(pdb_fn, uniprot):
    pdb_proc = import_pdb.PDB_Processer(['TPO'])
    chain_list = ['A', 'C']
    pdb = pdb_fn.split('.')[0]
    pdb_proc.process_pdb(pdb_fn, PDB_DIR, OUTPATH, uniprot)

    for chain in chain_list:
        comp = chain_obj(OUTPATH+pdb+'_'+chain+'.pdb', chain)
        ref = chain_obj(CHECKPATH+pdb+'_'+chain+'.pdb', chain)

        res_list_comp = [list(res.id)[1] for res in comp.get_residues()]
        res_list_ref = [list(res.id)[1] for res in ref.get_residues()]

        assert all(res_list_comp == res_list_ref)


# def test_dry_apo_pdb():
#     filename = '4EOQ.pdb'
#     pdb = '4EOQ'
#     chain = 'A'
#     parse = PDBParser()
#     structure = parse.get_structure(pdb, file=PDB_DIR+filename)[0]
#     NCAA = ('TPO', 'ASD')
#
#     io = PDBIO()
#     io.set_structure(structure[chain])
#     io.save("%s/%s_%s.pdb"%(OUTPATH,pdb,chain), select=dry_apo_pdb(*NCAA))


# @pytest.mark.parametrize("pdb_fn, res, value", zip(TEST_PDB_LIST+MISNUMBERED_PDB, \
#     [10,10,672], [(None,None), (None,None), (672,696)]))
# def test_check_one_SIFTS_res(pdb_fn, res, value):
#     pdb = pdb_fn.split('.')[0]
#     assert check_one_SIFTS_res(pdb, 'A', res) == value
#
# @pytest.mark.parametrize("pdb_fn",[MULTIPLE_CHAIN_PDB[1]])
# def test_query_SIFTS(pdb_fn):
#     query_SIFTS_info(pdb_fn)
#
# def test_check_start_stop_SIFTS_res():
#     assert check_start_stop_SIFTS_res('6GUE', 'A', 1, 298) == True
#     # assert check_start_stop_SIFTS_res('4jrv', chain='A') == False
