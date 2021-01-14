import pytest
from numpy import array_equal
from dist_analy import determine_seq
from Bio import AlignIO

PDB_LIST = ['4EOQ.pdb']
CHAIN_LIST = [('A', 'C')]
UNIPROT_LIST = [ 'P24941', 'P11802', 'Q00534']
PROT_LIST = ['CDK2', 'CDK4', 'CDK6']
@pytest.mark.parametrize("pdb, chain_list", zip(PDB_LIST, CHAIN_LIST))
def test_get_klifs_res(pdb, chain_list):
    for chain in chain_list:
        assert array_equal(determine_seq.get_klifs_res(pdb, chain), [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 33, 34, 35, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 143, 144, 145, 146, 147, 148, 149])

@pytest.mark.parametrize("unip_list, prot_list", zip([UNIPROT_LIST], [PROT_LIST]))
def test_get_and_align(unip_list, prot_list):
    align_ref = AlignIO.read("./datafiles/align/ref/cdk246_ref.al", "clustal")
    align = determine_seq.get_and_align_sequence("cdk246", "./datafiles/", unip_list, prot_list=prot_list )
    for seq1, seq2 in zip(align, align_ref):
        assert array_equal(seq1.seq, seq2.seq)

@pytest.mark.parametrize("s1,s2, ans", zip([[0,1,2,3]], [[1,3,4,5,7]], [[0,1,2,3,4,5,7]]))
def test_def_union(s1,s2, ans):
    union, union_ind = determine_seq.def_union(s1,s2)
    assert array_equal(union, ans)

CDK1_20_UNIP = ['P21127','Q9BWU1','Q96Q40','Q00526','Q9NYV4','P50750','P50613','Q00534','Q00536','P49336','Q15131','Q00535','Q9UQ88','P11802',\
 'P06493','Q8IZL9','Q14004','Q07002','P24941','O94921','Q00537']
CDK1_20 = ['CD11B_HUMAN','CDK19_HUMAN','CDK15_HUMAN','CDK3_HUMAN', 'CDK12_HUMAN','CDK9_HUMAN','CDK7_HUMAN', 'CDK6_HUMAN', \
 'CDK16_HUMAN','CDK8_HUMAN', 'CDK10_HUMAN','CDK5_HUMAN', 'CD11A_HUMAN','CDK4_HUMAN', 'CDK1_HUMAN', 'CDK20_HUMAN','CDK13_HUMAN','CDK18_HUMAN',\
 'CDK2_HUMAN','CDK14_HUMAN','CDK17_HUMAN']
CDK2_KLIFS_IDENT = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 33, 34, 35, 43, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 114, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 139, 142, 143, 144, 145, 146, 147, 148, 149, 163, 165, 167, 168, 169, 171, 174, 185, 187, 190, 192, 195, 220, 227, 262, 274]
@pytest.mark.parametrize("unip_list, prot_list", zip([CDK1_20_UNIP], [CDK1_20]))
def test_all(unip_list, prot_list):
    alignment = determine_seq.get_and_align_sequence('CDK1_20', './datafiles/', unip_list, prot_list=prot_list)
    cons_id = determine_seq.get_conserved(alignment)
    klifs_res = determine_seq.get_klifs_res('3BHT','A')
    cdk2_klifs_ident, union_id = determine_seq.def_union(cons_id[prot_list.index('CDK2_HUMAN')], klifs_res)
    assert array_equal(cdk2_klifs_ident, CDK2_KLIFS_IDENT)
