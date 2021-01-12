import numpy as np
import pytest
from dist_analy.util import pdb_info


TEST_PDB_LIST = ['2VU3.pdb', '5OO0.pdb']
TEST_CHAIN = ['A', 'A']
MULTIPLE_CHAIN_PDB = ['2WPA.pdb','4EOQ.pdb', '6GUE.pdb']
MULTIPLE_CHAIN = ['A,C', 'A,C', 'A']

UNIPROT_CDK2 = ['P24941']
UNIPROT_SEQ = ["MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNH\
PNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHS\
HRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYY\
STAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSF\
PKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"]


@pytest.mark.parametrize("pdb_fn, chain",zip(TEST_PDB_LIST+MULTIPLE_CHAIN_PDB, TEST_CHAIN+MULTIPLE_CHAIN))
def test_get_any_info_1(pdb_fn, chain):
    pdb = pdb_fn.split('.')[0]
    uniprot_url = 'https://data.rcsb.org/rest/v1/core/uniprot/'
    entity_url = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'
    assert pdb_info.get_any_info(uniprot_url, pdb, str(1))[0]['rcsb_uniprot_container_identifiers']['uniprot_id'] == 'P24941'
    assert pdb_info.get_any_info(entity_url, pdb, str(1))['entity_poly']['pdbx_strand_id'] == chain

@pytest.mark.parametrize("uniprot, seq",zip(UNIPROT_CDK2, UNIPROT_SEQ))
def test_get_any_info_2(uniprot, seq):
    seq_url = 'http://www.ebi.ac.uk/proteins/api/proteins/'
    assert pdb_info.get_any_info('http://www.ebi.ac.uk/proteins/api/proteins/', uniprot)['sequence']['sequence'] == seq

def test_get_any_info_3():
    request_dict = {
        "kinase_group": "CMGC",
        "kinase_family": "CDK",
        "species": "Human"
    }

    kinase_names = pdb_info.get_any_info('https://klifs.net/api/kinase_names', **request_dict)
    kinase_id = set([temp_dict['kinase_ID'] for temp_dict in kinase_names])
    assert kinase_id == set(np.append(np.arange(196,216), 1103))
