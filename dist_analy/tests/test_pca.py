import numpy as np
import pytest
import dist_analy.pca
import dist_analy.dist_analy


"""
TODO: Create more robust testing of pca -> will be useful when implementing classes
"""
DIST_MAT_DIR = './datafiles/npy_files/calc/'
SMALL_DIST_MAT = np.array([[[0, 1, 2, 3], [10, 0 , 12, 13], [20, 21, 0, 23], [30, 31, 32, 0]], \
                        [[0, 0, 2, 3], [0, 0 , 0, 0], [20, 0, 0, 23], [30, 0, 32, 0]], \
                        [[0, 1, 2, 3], [10, 0, 12, 13], [20, 21, 0, 23], [30, 31, 32, 0]]])

SMALL_DIST_MAT_1 = np.array([[[0, 1, 2, 3], [10, 0 , 12, 13], [20, 21, 0, 23]], \
                        [[0, 0, 2, 3], [0, 0 , 0, 0], [20, 0, 0, 23]], \
                        [[0, 1, 2, 3], [10, 0, 12, 13], [20, 21, 0, 23]]])

SMALL_SQUARE_DIST_MAT = np.array([[[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], \
                        [[0, 1, 0, 3], [1, 0, 0, 5], [0, 0, 0, 0], [3, 5, 0, 0]], \
                        [[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]]])

def test_remove_missing():
    res_list = [7,8,9,10]
    mat_out, res_cons, ind_cons = dist_analy.pca.remove_missing_(SMALL_DIST_MAT, res_list)
    assert ind_cons == [0,2,3]

def test_run_unsym():
    res_list = [7,8,9,10]
    with pytest.raises(ValueError) as e:
        dist_analy.pca.run(SMALL_DIST_MAT_1, res_list, 4)
    assert str(e.value) == 'PCA calculates symetric distance matrices'

@pytest.mark.parametrize("feats", [np.array([[1,1], [2.2,2], [2.3,1], [1.5,1.5]])])
@pytest.mark.parametrize("ans", [[3,0,1,2]])
def test_find_medoid(feats, ans):
    print(np.equal(dist_analy.pca.find_medoid(feats,np.arange(0,4)), ans))
    assert np.equal(dist_analy.pca.find_medoid(feats,np.arange(0,4)), ans).all()

@pytest.mark.parametrize("feats", [SMALL_DIST_MAT])
def test_triu_flatten(feats):
    assert np.array_equal(dist_analy.pca.triu_flatten(feats[0], np.arange(7,11)), np.array([1,2,3,12,13,23]))
    assert np.array_equal(dist_analy.pca.triu_flatten(feats, np.arange(7,11)), np.array([[1,2,3,12,13,23], [0,2,3,0,0,23],[1,2,3,12,13,23]]))

def test_get_indices():
    fc = [1,1,2,2,2,1,3,3]
    ind_hier = [0,1,5,2,3,4,6,7]
    assert np.array_equal(dist_analy.pca.get_indices(fc, ind_hier, [1]), [0,1,5])
    assert np.array_equal(dist_analy.pca.get_indices(fc, ind_hier, [1,2]), [0,1,5,2,3,4])
# @pytest.mark.parametrize("dist_mats", SMALL_DIST_MAT)
# def test_pca(dist_mats):
#     res_list = [7,8,9,10]

@pytest.mark.parametrize("feats", [SMALL_DIST_MAT[0]])
def test_triu_getXY(feats):
    k=1
    m=len(feats[0])
    test_ind = [0,1,4]
    ans = [(0,1), (0,2), (1,3)]
    for i,(x1,y1) in zip(test_ind, ans):
        x2,y2 = dist_analy.pca.triu_getXY(i,m,k)
        assert feats[x2][y2] == feats[x1][y1]

    k=0
    ans = [(0,0), (0,1), (1,1)]
    for i,(x1,y1) in zip(test_ind, ans):
        x2,y2 = dist_analy.pca.triu_getXY(i,m,k)
        assert feats[x2][y2] == feats[x1][y1]

@pytest.mark.parametrize("feats", [SMALL_DIST_MAT[0]])
def test_triu_getXY_max_ind(feats):
    k=1
    m=len(feats[0])

    with pytest.raises(ValueError) as e:
        x2,y2 = dist_analy.pca.triu_getXY(7,m,k)
    assert str(e.value) == 'i is greater than the max triangular index'

    x2,y2 = dist_analy.pca.triu_getXY(6,m,k)
    print(x2,y2)
    assert feats[x2][y2] == feats[2][3]

@pytest.mark.parametrize("mats", [SMALL_SQUARE_DIST_MAT ])
def test_replace_zeros(mats):
    feats = dist_analy.pca.triu_flatten(mats, np.arange(7,11))
    feats_mean = dist_analy.pca.replace_zeros(feats, method='mean')
    feats_median = dist_analy.pca.replace_zeros(feats, method='median')
    ans_array = np.array([mats[0],mats[0],mats[0]])
    ans = dist_analy.pca.triu_flatten(ans_array, np.arange(7,11))
    # print(feats_mean, ans_array, ans)
    assert np.array_equal(feats_mean, ans)
    assert np.array_equal(feats_median, ans)

    print(dist_analy.pca.replace_zeros(mats, method='mean'))
