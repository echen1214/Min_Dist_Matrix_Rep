import warnings
import numpy as np
import scipy.cluster.hierarchy as sch
import sklearn.cluster as skc
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from collections import Counter
from sklearn import decomposition
from colored import fg, bg, attr
from copy import copy
from matplotlib.lines import Line2D
from IPython.display import Markdown
from IPython.display import display

""" TODO
- organize data properly
- x calculate principal compenents
    - x output PCA plot/ eigenvalue contributions
    - create interactive plot where you can interactively select ranges in jupyter
      notebook and it'll return the PDB
- x determine the important distances - Z-score
    - create interactive plot where you can interactively select ranges in jupyter
      notebook and it'll return the PDB
- cluster by PCA
- apply checks to make sure parameters are correct:
    -> make the function take flexible length lists
    - plot_r1r2 -> check if family, pdb_prot_index, dist_mats are same lengths
    - plot_stacked_histogram
    -> make more consistent on what matrix input (distance matrix vs flattened)
        is needed so that it is easy to use
    pca -> requires flattened arrays
    r1r2/ stacked histo -> requires distance matrix
make class
attribute list:
    dist_mat
    res_list
    dist_mat_cons
    res_list_cons
    ind_list_cons
    feats
    inds_fc
    npy_pca
"""
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

COLOR_LIST = ['g','r','c','m','y','k','orange', 'pink', 'grey', 'lime', 'tan', 'aqua', 'olive', 'dodgerblue']
MARKER_LIST = ["o", "v", "s", "P", "*", "X", "d", ">", "2", "p", "+", 'x', '<', '|', '_', 'h', '8', 'H']
DFLT_COL = "#808080"

def hist_missing_residue(dist_mats: np.ndarray, res_list: list, res_get: list = None,\
                         axis: int = 1):
    """Plot the frequency that a residue is missing across a set of residue–residue
    distance matrices. Returns a list of structure indices that are missing residues
    in the list res_get

    Parameters
    ----------
    dist_mats : np.ndarray
        array of distance matrices (3D: PDB * res_list * res_list).
    res_list : list
        list of residues corresponding to rows and columns of the distance matrix
    res_get : list, optional
        list of residues to track which structure indices are missing these residues
    axis : int, default: 1
        axis on which to check for missing residues

    Returns
    -------
    list
        list of structure indices that are missing residues in the list res_get

    """
    if not res_get:
        res_get = []

    plt.figure(figsize=(len(res_list)*0.15,5))
    missing_res = [0 for _ in range(len(res_list))]
    # struct_res_get = [0 for _ in range(len(res_get))]
    missing_pdb = []
    for i,mat in enumerate(dist_mats):
        missing = np.where(~mat.any(axis=axis))[0]
        for j in missing:
            missing_res[j] += 1
            if res_list[j] in res_get:
                missing_pdb.append(i)
    plt.bar(range(len(res_list)), missing_res)
    plt.xticks(range(0,len(res_list),2), labels=res_list[::2], rotation=90)
    plt.ylim([0, len(dist_mats)])
    # plt.title("Frequency of missing KLIFS-IDENT residues")
    plt.ylabel("Number of structures")
    plt.xlabel("Residue ID")
    plt.margins(0)
    return (missing_pdb, missing_res)

def hist_missing_structure(dist_mats: np.ndarray, cutoff: int = 10, bins: int = None, \
                           axis: int = 1):
    """Plot a histogram of the frequency of residues missing per structure based
    on the distance matrices. Returns list of struture indices that have more residues
    missing than the cutoff

    Parameters
    ----------
    dist_mats : np.ndarray
        array of distance matrices (3D: PDB * res_list * res_list).
    cutoff : int, default: 10
        Cutoff of residues missing
    bins : int, optional
        Number of bins to plot the histogram
    axis : int, default: 1
        axis on which to check for missing residues
    Returns
    -------
    list
        list of struture indices that have more residues missing than the cutoff

    """
    missing_pdb = []
    many_missing = []

    for i,mat in enumerate(dist_mats):
        missing = np.where(~mat.any(axis=axis))[0]
        missing_pdb.append(len(missing))
        if len(missing) > cutoff:
            many_missing.append(i)
    plt.figure()
    plt.hist(missing_pdb, bins=bins)
    # plt.title("Frequency of missing KLIFS-IDENT residues")
    plt.ylabel("Number of structures")
    plt.xlabel("Number of missing residues")
    return many_missing


def remove_missing_(dist_mats: np.ndarray, res_list: list, inf: bool = True):
    """ Pass in a list of the distance matrices and the corresponding residue lists,
    determine the subset of residues that are present in every distance matrix

    Parameters
    ----------
    dist_mats : np.ndarray
        array of distance matrices (3D: PDB * res_list * res_list).
    res_list : list
        list of residues corresponding to rows and columns of the distance matrix
    inf : bool
        flag to remove infinite values, default is to remove values of 0
        
    Returns
    -------
    dist_mats_cons : np.ndarray
        array of distance matrices with subset of residues
        (3D: PDB * res_list * res_list)
    res_cons: list
        list of the subset residue IDs
    ind_cons: list
        list of the subset residue indices of the original residue list

    """

    missing_res = Counter()

    for mat in dist_mats:
        if inf:
            missing = np.where((mat==np.inf).all(axis=1) | (~mat.any(axis=1)))[0]
        else:
            missing = np.where(~mat.any(axis=1))[0]
        missing_res.update(missing)

    print("original length of residue list %i; remove %i residues"%\
         (len(res_list),len(missing_res.keys())))
    ind_cons = [x for x in range(len(res_list)) if x not in missing_res]

    res_cons= [res_list[x] for x in range(len(res_list)) if x not in missing_res]

    # print(len(ind_cons), res_cons)

    dist_mats_cons = []
    for mat in dist_mats:
        dist_mats_cons.append(mat[np.ix_(ind_cons, ind_cons)])
    return(np.array(dist_mats_cons), res_cons, ind_cons)

def replace_zeros_(feats: np.ndarray, method:str ='mean', axis: int = 0):
    """Replaces the zeros of list of distance matrix or feeatuer matrix with
    either the mean or median of that particular distance

    Parameters
    ----------
    feats : np.ndarray
        array of features (2D: PDB * features) or distance matrices (3D: PDB *
        res_list * res_list). axis 0 must be the array of structures
    method : str, default: 'mean'
        string 'mean' or 'median'
    axis : int, default: 0
        axis to perform the mean/median replacement

    Returns
    -------
    np.ndarray
        features or distance matrices where the 0s are replaced by the 'mean' or
        'median' value along axis = 0.
        if distance matrices are given, then the 0s along the diagonal will be
        replaced by a nan

    """
    if method == 'mean':
        new_feats = np.where(feats==0.0, np.nanmean(np.where(feats==0.0, np.nan, feats), axis = axis), feats)
    if method == 'median':
        new_feats = np.where(feats==0.0, np.nanmedian(np.where(feats==0.0, np.nan, feats), axis = axis), feats)
    return new_feats

def triu_flatten(dist_mats: np.ndarray, len_res_list: int, k: int = 1):
    """ Return an array of flattened 1D upper triangular values (features) of each
    symetric residue-residue distance matrix. These values include all those above the
    diagonal of the matrix (k=1)

    Parameters
    ----------
    dist_mats : np.ndarray
        array of distance matrices (3D: PDB * res_list * res_list | 2D: res_list * res_list)
    len_res_list : list
        number of elements of res_list
    k : int, default = 1
        Diagonal offset. Default is 1, removing the diagonal entires. k = 2 can be
        used to remove adjacent residues if the res_list is consecutive
    Returns
    -------
    np.ndarray
        array of 1D flattened distance matrices (2D: PDB * features | 1D: res_list)

    """
    triu_ind = np.triu_indices(len_res_list, k=k)
    feats_list = []
    # print(dist_mats.shape)
    if len(dist_mats.shape) == 3:
        for mat in dist_mats:
            feats_list.append(np.array(mat)[triu_ind])
    elif len(dist_mats.shape) == 2:
        feats_list = np.array(dist_mats)[triu_ind]
    feats_list=np.array(feats_list)
    return(feats_list)

def get_indices(fc: np.ndarray ,ind_hier: list, val: list):
    """ Gets the indices of leaf nodes that correspond to the specified cluster

    Parameters
    ----------
    fc : np.ndarray
        flat cluster from hierachical clustering, sch.fcluster().
        fc[i] is the cluster number that i belongs to
    ind_hier : list
        list of integer labels corresponding to the leaf nodes from sch.dendrogram()
    val : list
        list of subset cluster numbers

    Returns
    -------
    list
        list of leaf nodes for the subset of cluster

    """
    out = []
    for inv in ind_hier:
        for x in val:
            if fc[inv]==x: out.append(inv)
    return out

def feat_dist_matrix(feats: np.ndarray):
    """Calculate a pairwise distance matrix between each 1D feature array

    Parameters
    ----------
    feats : np.ndarray
        array of 1D flattened distance matrices

    Returns
    -------
    np.ndarray
        Distance matrix

    """
    d_mat = []
    for x in feats:
        d_temp = []
        for y in feats:
            d_temp.append(np.linalg.norm(x-y))
        d_mat.append(d_temp)
    return (np.array(d_mat))

def find_medoid(feats: np.ndarray, ind_map: list):
    """Determine the medoid structure of a cluster by calculating the pairwise
    euclidean distances between each of the features and choosing the structures
    based on lowest summed pairwise distances. The rest of the structures are
    sorted based on lowest distance to the medoid

    Parameters
    ----------
    feats : np.ndarray
        array of 1D flattened distance matrices
    ind_map : list
        list of indices corresponding to the structures of a cluster.

    Returns
    -------
    list
        sorted list of indices corresponding to the structures of the cluster
        from medoid to furthest from medoid

    """

    if len(feats.shape) == 1:
        return ind_map
    d_mat = feat_dist_matrix(feats)
    ## rank by closeness to medoid
    med_ind = np.argmin(d_mat.sum(axis=0))
    # print("med_ind", med_ind)
    sort_ind = np.argsort(d_mat[med_ind])
    return ([ind_map[x] for x in sort_ind])

def clustering(feats: np.ndarray, k: int, npy_pca, method: str = 'ward', criterion: str = 'maxclust'):
    """Performs heirarchcial clustering to create k clusters. Plots out dendrogram
    and colors the branches based on hierarchical cluster. inds_fc and
    medoid_ind_list are not equivalent

    The cluster colors are green, red, cyan, maroon, yellow, black. Any unclusted
    structures are colored a default color, grey.

    Parameters
    ----------
    feats : np.ndarray
        array of 1D flattened distance matrices
    k : int
        number of clusters
    method : str, default: 'ward'
        method options, https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    criterion : str, default: 'maxclust'
        criterion option, https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html

    Returns
    -------
    inds_fc : list
        2D list of structure indices for each cluster (2D: cluster * structure indices)
    org_color_dict : dict
        dictionary of the cluster color of each structure index. org_color_dict[i]
        is the cluster color of structure i
    medoid_ind_list : list
        2D list of sorted structure indices for each cluster (2D: cluster * structure indices)

    """
    Z = sch.linkage(feats,method=method)
    fc = sch.fcluster(Z,t=k,criterion=criterion)

    ct=Z[-(k-1),2]
    R = sch.dendrogram(Z, no_plot=True, leaf_rotation=90., color_threshold=ct)
    ind_map = [int(x) for x in R['ivl']]

    medoid_ind_list = []

    inds_fc = []
    org_color_dict=dict()
    index = 0
    for i in range(1,k+1):
        ind_fc = get_indices(fc,ind_map,[i])
        if len(ind_fc)==1:
            org_color_dict[ind_fc[0]]=DFLT_COL
        else:
            for x in ind_fc:
                org_color_dict[x]=(COLOR_LIST[index])
            index=index+1

        ind_fc_clust = [j for j,x in enumerate(ind_fc)]
        medoid_ind=find_medoid(feats[ind_fc], ind_fc_clust)

        inds_fc.append(ind_fc)
        medoid_ind_list.append(medoid_ind)
#    print("link_cols")
    link_cols = {}
    ct=Z[-(k-1),2]
    for i, il2 in enumerate(Z[:,:2].astype(int)):
        if il2[0] > len(Z):
            c1 = link_cols[il2[0]]
        else:
            c1 = org_color_dict[il2[0]]

        if il2[1] > len(Z):
            c2 = link_cols[il2[1]]
        else:
            c2 = org_color_dict[il2[1]]

        if c1 == c2:
            link_cols[i+1+len(Z)] = c1
        else:
            link_cols[i+1+len(Z)] = DFLT_COL
#    print("dendrogram")
    plt.figure(figsize=(25, 5))
    R = sch.dendrogram(Z, no_plot=False,leaf_rotation=90., color_threshold=None,\
                        link_color_func=lambda x: link_cols[x], no_labels=True)
    return (inds_fc, org_color_dict, medoid_ind_list)

def plot_analy(npy_pca: decomposition.PCA, feats: np.ndarray, k: int, \
               family_map: list = None, family: list = None):
    """Plots the dendrogram and the PC1/PC2 plot
    Parameters
    ----------
    npy_pca : sklearn.decomposition.PCA
        PCA object
    feats_list : np.ndarray
        array of 1D flattened distance matrices
    k : int
        number of clusters
    family_map: list, optional
        flat list mapping each distance matrix to the index of family list
    family: list, optional
        list of labels

    """
    inds_fc, org_color_list, medoid_ind_list = clustering(feats, k, npy_pca)
    plot_pca(npy_pca, feats, inds_fc, family_map, family)
    return(inds_fc, medoid_ind_list)

def get_pca(feats: np.ndarray, cumsum: float = 0.8):
    """Centers and fits PCA object to data and returns the projected coordinates

    Parameters
    ----------
    feats : np.ndarray
        array of 1D flattened distance matrices
    """

    npy_pca = decomposition.PCA(n_components = 10, svd_solver = 'full', random_state=42)
    a = npy_pca.fit_transform(feats)

    var_ratio = npy_pca.explained_variance_ratio_
    sel_axis = np.where(np.cumsum(var_ratio) < cumsum)[0]
    if len(sel_axis) < 2:
        sel_axis=[0,1]
    elif len(sel_axis) >= 10:
        sel_axis=range(0,10)
    else:
        sel_axis = np.append(sel_axis, sel_axis[-1]+1)
    sel_var = [f'{var:.2f}' for var in var_ratio[sel_axis]]
    print(f"Cumulative explained variance > {cumsum}: {', '.join(sel_var)}")
    proj_coords = a[:,sel_axis]
    return proj_coords, var_ratio[sel_axis], npy_pca

def pca_hdbscan(proj_coords, var_ratio, hdbscan_args: dict = dict(), family_map: list = None, family: list = None):
    hdb = skc.HDBSCAN(**hdbscan_args).fit(proj_coords)
    pca_hdbscan_figure(proj_coords,hdb.labels_, hdb.probabilities_, var_ratio, family_map, family)
    return(hdb.labels_, hdb)

def pca_hdbscan_figure(a, labels: list, prob: list, var_ratio: list, family_map: list = None, family: list = None, resize_by_prob=False):
    ## inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html#demo-of-hdbscan-clustering-algorithm

    plt.figure()
    unique_labels = set(labels)
    proba_map = {idx: prob[idx] for idx in range(len(labels))}

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    plt.title(f"Estimated number of clusters: {n_clusters_}")

    for lab in unique_labels:
        if lab == -1:
            # Black used for noise.
            col = "black"
            # continue
        else:
            col = COLOR_LIST[lab]

        label_index = np.where(labels == lab)[0]
        # for ci in class_index:
        if family:
            for mark in range(min(family_map), max(family_map)+1):
                label_fam_index = np.intersect1d(label_index, np.where(np.asarray(family_map)==mark)[0])
                if resize_by_prob:
                    for ci in label_fam_index:
                        plt.plot(a[ci, 0],a[ci, 1],
                            "x" if lab == -1 else MARKER_LIST[mark], markerfacecolor=col,
                            markeredgecolor="k", markersize=5 if lab == -1 else 1 + 5 * proba_map[ci])
                else:
                    plt.plot(a[label_fam_index, 0],a[label_fam_index, 1],
                        "x" if lab == -1 else MARKER_LIST[mark], markerfacecolor=col,
                        markeredgecolor="k",)
        else:
            if resize_by_prob:
                for ci in label_index:
                    plt.plot(a[ci, 0],a[ci, 1],
                        "x" if lab == -1 else "o", markerfacecolor=col,
                        markeredgecolor="k", markersize=5 if lab == -1 else 1 + 5 * proba_map[ci])
            else:
                plt.plot(a[label_index, 0],a[label_index, 1],
                    "x" if lab == -1 else "o", markerfacecolor=col,
                    markeredgecolor="k",)
    
    plt.xlabel(f"PC1: {var_ratio[0]*100:.1f}")
    plt.ylabel(f"PC2: {var_ratio[1]*100:.1f}")
    if family:
        legend_elements = [Line2D([], [], marker=mark, label=label, color='k', markerfacecolor='white', linestyle='None') for mark, label in zip(MARKER_LIST[:len(family)], family)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        pass

# assumes square matrix
# does not work for negative k
# m is one dimension of square matrix
def triu_getXY(i: int, m: int, k:int):
    """Returns the corresponding X,Y value of a 2D square matrix of size m * m
    given the upper triangular (triu) index and the diagonal k.

    Parameters
    ----------
    i : int
        triu index
    m : int
        dimension of one axis of the 2D square matrix
    k : int
        diagonal which element below are zerod

    Returns
    -------
    tuple
        corresponding X,Y value

    """
    if k < 0:
        raise ValueError('K is a negative number. 0 < k < m')
    if k >= m:
        raise ValueError('K is greater than or equal to m. 0 < k < m')
    l = m-k
    max_ind = l*(l+1)/2 #nth triangular numbers
    if i > max_ind:
        raise ValueError('i is greater than the max triangular index')
    x = 0

    while (i//l != 0):
        i -= l
        l -= 1
        if l == 0:
            break
        x += 1

    if l == 0:
        y=x+k
    else:
        y = (i%l)+(x+k)
    return(x, y)

def calc_cluster_smd(cluster1_inds: list, cluster2_inds: list, feats: np.ndarray, \
                        norm: int = 1.5, std: str = "SMD"):
    """Calculate the standardized mean difference (SMD) or the strictly standardized
    mean difference (SSMD) of the pairwise residue-residue distances (features)
    between two clusters. The positive SMD values reflect distance pairs that are
    shorter for cluster2_ind, whereas the negative SMD values reflect the short
    cluster1_ind pairs.

    SMD = average(cluster 1 features) - average(cluster 2 features) / (sqrt (std(cluster 1 features) * std(cluster 2 features)))
    SSMD = average(cluster 1 features) - average(cluster 2 features) / (sqrt (std(cluster 1 features)^2 + std(cluster 2 features)^2))
    This value is then biased by dividing by (minimum feature value - 1.5)

    Parameters
    ----------
    cluster1_inds : list
        list of indices of the feats array that reflect cluster 1
    cluster2_inds : list
        list of indices of the feats array that reflect cluster 2
    feats : np.ndarray
        array of 1D flattened distance matrices
    norm : int, default: 1.5
        SMD is normalized by dividing by (min_dist - norm). If your dataset
        contains hydrogen atoms set norm to 0
    std : str, default: 'SMD'
        'SMD' or 'SSMD' string reflecting the choice of denominator

    Returns
    -------
    tuple
        the biased SMD value, minimum distance list

    """

    if std not in ['SMD', 'SSMD']:
        raise ValueError('std must be either standardized mean difference `SMD` or \
                          or strictly standardized mean difference `SSMD`')
    mean_feats_c1 = np.average(feats[cluster1_inds], axis=0)
    mean_feats_c2 = np.average(feats[cluster2_inds], axis=0)

    std_feats_c1 = np.std(feats[cluster1_inds], axis=0)
    std_feats_c2 = np.std(feats[cluster2_inds], axis=0)

    num = np.subtract(mean_feats_c1, mean_feats_c2)
    # print(num)
    if std == 'SSMD':
        denom = np.sqrt(np.add(np.square(std_feats_c1),np.square(std_feats_c2)))
    if std == 'SMD':
        std_feats_c1= np.where(std_feats_c1!=0, std_feats_c1, 1)
        std_feats_c2= np.where(std_feats_c2!=0, std_feats_c2, 1)
        denom = np.sqrt(np.multiply(std_feats_c1,std_feats_c2))
    # print(denom)
    smd_diff_c1c2 = np.divide(num, denom)
    # zscore_diff_c1c2_idx = np.argsort(zscore_diff_c1c2)[::-1]

    concat_ind = cluster1_inds + cluster2_inds
    min_dist = np.amin([feats[x] for x in concat_ind], axis=0)
    smd_new = np.divide(smd_diff_c1c2, (np.subtract(min_dist,norm))) #np.divide(np.log(min_dist), np.log(5)))
    return (smd_new, min_dist)

def plot_smd_distrib(cluster1: int, cluster2: int, feats: np.ndarray, min_feats: np.ndarray, \
                        smd: np.ndarray, xcutoff: int=3.5, ycutoff: int=5, \
                        norm: int = 1.5, std: str = "SMD"):
    """Plots the distribution of SMD vs minimum distance. Each data point
    represented one distance pair

    Can also plot distribution of SSMD vs minimum distance.

    Parameters
    ----------
    cluster1 : int
        cluster index number
    cluster2 : int
        cluster index number
    feats : np.ndarray
        array of 1D flattened distance matrices
    smd : np.ndarray
        array of SMD/SSMD values for each distance pair
    xcutoff : int
        distance cutoff value
    ycutoff : int
        SMD cutoff value
    norm : int, default: 1.5
        SMD is normalized by dividing by (min_dist - norm). If your dataset
        contains hydrogen atoms set norm to 0
    std : str, default: 'SMD'
        'SMD' or 'SSMD' string reflecting the choice of denominator
    # excl_adj : bool, default: 'False'
    #     if True, excludes any residue pairs that are adjacent to one another
    """
    # min_feats = np.amin(feats, axis=0)
    # min_feats_idx = np.argsort(min_feats)
    # print(min_feats.shape, smd.shape)
    min_smd= np.stack((min_feats,smd),axis=1)
    plt.figure()
    plt.scatter(min_smd[:,0],min_smd[:,1])
    plt.xlabel("Minimum distance pair across all structures")
    plt.ylabel("%s$^{%i|%i}}$(min_dist-%.2f)"%(std, cluster1, cluster2, norm))
    plt.axvline(x=xcutoff, color="red")
    plt.axhline(y=0, color='black')
    plt.axhline(y=ycutoff, color='red', linestyle='dashed')
    plt.axhline(y=-ycutoff, color='red', linestyle='dashed')
    if xcutoff > 10:
        plt.xlim(-0.5,xcutoff)
    else:
        plt.xlim(-0.5, 10)

###, color_list, region_list):
class color_text:
    def __init__(self, color_list: list, region_list: list):
        self.color_list = color_list
        self.region_list = region_list
    def get_text(self, res: str, res_num: int):
        """Creates colored string corresponding to the specified regions and region
        colors using the colored package

        Parameters
        ----------
        res : str
            one letter residue code
        res_num : int
            residue ID
        Returns
        -------
        str
            colored string of the one letter code and residue ID

        """
        # if res_num in region_list[2]:
        #     return(fg("%s"%color_list[2])+res+str(res_num)+attr(0))
        for i,reg in enumerate(self.region_list):
            if res_num in reg:
                return(fg("%s"%self.color_list[i])+res+str(res_num)+attr(0))
        return res+str(res_num)
    def get_text_markdown(self, res:str, res_num: int):
        for i,reg in enumerate(self.region_list):
            if res_num in reg:
                return("<span style='color: %s'>"%self.color_list[i] + "%s"%(res+str(res_num)) + "</span>")
        return res+str(res_num)
    def return_color(self, res_num):
        for i,reg in enumerate(self.region_list):
            if res_num in reg:
                return(self.color_list[i])
        return("black")
# def color_text(res: str, res_num: int, color_list: list, region_list: list):
#
#     # CDK2
#     # color_list  = ['red', 'orange_red_1', 'yellow', 'green_1', 'dark_green', 'cyan', 'blue', 'purple_1b', 'magenta'] ## change to hex code
#     # region_list = [np.arange(12,17), np.delete(np.arange(45,58), np.where(np.arange(45,58)==51)), \
#     #               np.arange(80,87), np.delete(np.arange(125,132), np.where(np.arange(125,132)==127)), \
#     #                np.array([33,51,127,145]), np.arange(146,172),np.arange(182,197)]
#
#     # ABL1
#     # region_list = [np.arange(248, 256), np.arange(282,292),[271, 286, 344, 381],np.arange(317,322),\
#     #             np.arange(338, 345),np.arange(380, 403),[]]
#
#     if res_num in region_list[2]:
#         return(fg("%s"%color_list[2])+res+str(res_num)+attr(0))
#     for i,reg in enumerate(region_list):
#         if res_num in reg:
#             return(fg("%s"%color_list[i])+res+str(res_num)+attr(0))
#     return res+str(res_num)

def plot_smd(cluster1: int, cluster2: int, feats: np.ndarray, min_dist: np.ndarray, \
                smd: np.ndarray, res_list_list: list, uniprot_seq_list: list, color_obj_list: list = None, \
                prot_list: list = None, \
                top: int = 10, xcutoff:int = 3.5, ycutoff:int = 5, norm: int = 1.5, \
                std: str = 'SMD', k: int = 1):
    """ Plots SMD vs minimum distance. Returns the residue–residue ID of the top SMD ranking distance pairs
    given a minimum distance cutoff (xcutoff) and a minimum absolute SMD
    value (ycutoff).

    Can also plot SSMD vs minimum distance

    SMD is normalized by dividing by (min_dist - norm)

    Parameters
    ----------
    cluster1 : int
        cluster index number
    cluster2 : int
        cluster index number
    feats : np.ndarray
        array of 1D flattened distance matrices
    min_dist : np.ndarray
        array of minimum distance for each residue–residue pair
    smd : np.ndarray
        array of smd values for each distance pair
    res_list : list
        list of residues corresponding one axis of the distance matrix
    uniprot_seq : str
        string of one letter amino acid codes that reflect the sequence which
        the residue numbering is based off of
    xcutoff : int
        distance cutoff value
    ycutoff : int
        smd cutoff value
    norm : int, default: 1.5
        smd is normalized by dividing by (min_dist - norm). If your dataset
        contains hydrogen atoms set norm to 0
    std : str, default: 'SMD'
        'SMD' or 'SSMD' string reflecting the choice of denominator
    k : int, default = 1
        Diagonal offset. Default is 1, removing the diagonal entires. k = 2 can be
        used to remove adjacent residues if the res_list is consecutive

    Returns
    -------
    list
        list of tuples containing the x index, y index, feature index, minimum
        distance, and SMD of the top distance pairs

    """
    # zscore_new_idx = np.argsort(zscore)[::-1]
    if std not in ['SMD', 'SSMD']:
        raise ValueError('std must be either standardized mean difference `SMD` or \
                          or strictly standardized mean difference `SSMD`')

    plot_smd_distrib(cluster1, cluster2, feats, min_dist, smd, \
                        xcutoff=xcutoff, ycutoff=ycutoff, norm=norm, std=std)

    # top_ind = 0
    feat_idx = []
    filter_cut = (abs(smd) > ycutoff) & (min_dist < xcutoff)
    filter_idx = np.where(filter_cut == True)[0]
    filter_z = smd[filter_idx]
    filter_dist = min_dist[filter_idx]

    z_pos = filter_z > 0
    z_neg = filter_z < 0
    # out_all = "|"
    # if prot_list:
    #     for i in prot_list:
    #         out_all += "%s|"%i
    #     out_all += "min dist (Å)| SMD|<br>|"
    #     for i in prot_list:
    #         out_all += ":-:|"
    #     out_all += ":-:| :-:|<br>"

    for z_rank, clust, rank in zip([z_pos, z_neg], [cluster2, cluster1], [-1,1]):
        print("cluster %i stabilizing interactions"%(clust+1))
        z_sort_idx = np.argsort(filter_z[z_rank])[::rank]
        idx_sort = filter_idx[z_rank][z_sort_idx]
        dist_sort = filter_dist[z_rank][z_sort_idx]
        z_sort = filter_z[z_rank][z_sort_idx]
        for i, (idx, min_dist, smd_val) in enumerate(zip(idx_sort, dist_sort, z_sort)):
            if i > top:
                break
            x,y=triu_getXY(idx,m=len(res_list_list[0]),k=k)
            feat_idx.append([x,y,idx,min_dist,smd_val])
            # out_text = "|"
            out_text = ""
            for i, (res_list, uniprot_seq) in enumerate(zip(res_list_list, uniprot_seq_list)):
                r1 = uniprot_seq[res_list[x]-1]
                r1_id = res_list[x]
                r2 = uniprot_seq[res_list[y]-1]
                r2_id = res_list[y]
                if color_obj_list:
                    out_text += "%s-%s: "%(color_obj_list[i].get_text_markdown(r1,r1_id), color_obj_list[i].get_text_markdown(r2,r2_id))
                    # out_text += "%s-%s|"%(color_obj_list[i].get_text_markdown(r1,r1_id), color_obj_list[i].get_text_markdown(r2,r2_id))
                    # out_text += "%s-%s: "%(color_obj_list[i].get_text(r1,r1_id), color_obj_list[i].get_text(r2,r2_id))
                else:
                    out_text += "%s-%s: "%(r1+str(r1_id), r2+str(r2_id))
                    # out_text += "%s-%s|"%(r1+str(r1_id), r2+str(r2_id))

            if np.isinf(smd_val):
                print("Warning: infinite value may result from std = 0")
                top_ind -= 1
            # if excl_adj:
            #     if abs(r2_id-r1_id) < 2:
        #         continue
            out_text += "%.3f, %.3f"%(min_dist,smd_val)
            display(Markdown(out_text))

            # out_text += "%.3f|%.3f| <br>"%(min_dist,smd_val)
            # out_all += out_text
        # display(Markdown(out_all))

            # print(out_text)
    return feat_idx

def plot_r1r2(c1: int, c2: int, r1r2_feat: list, labels: list, dist_mats: np.ndarray, \
    pdb_prot_index: list = None, family: list = None):
    """For each distance matrix plot on R2 vs R1. The R1/R2 distances are selected
    following the calculation of the SMD/SSMD. The top distances are passed in
    as r1r2_feat. The clustering is defined by inds_fc.

    To include multiple proteins in the R1/R2 plots, pass in two lists, pdb_prot_index
    and family. pdb_prot_index maps each distance matrix to the family.
    Parameters
    ----------
    c1 : int
        cluster index number 1
    c2 : int
        cluster index number 2
    r1r2_feat : list
        list of the top R1/R2 distances, and their corresponding minimum distance
        and SMD/SSMD values. Plots the list returned by plot_smd()
    inds_fc : list
        2D list of structure indices for each cluster (2D: cluster * structure indices)
    dist_mats : np.ndarray
        array of distance matrices (3D: PDB * res_list * res_list).
    pdb_prot_index : list
        list of family label indices corresponding to dist_mats
    family : list
        list of family labels
    """
    if not pdb_prot_index and family:
        raise ValueError("Must pass in both pdb_prot_index and family or neither")
    if not pdb_prot_index:
        pdb_prot_index = []
    if not family:
        family = []

    plt.figure()

    unique_labels = set(labels)
    r1_feat = np.asarray([r1r2 for r1r2 in r1r2_feat if r1r2[4] > 0])
    temp1 = np.sum(dist_mats[:, r1_feat[:,0].astype(int), r1_feat[:,1].astype(int)],axis=1)
    r2_feat = np.asarray([r1r2 for r1r2 in r1r2_feat if r1r2[4] < 0])
    temp2 = np.sum(dist_mats[:, r2_feat[:,0].astype(int), r2_feat[:,1].astype(int)],axis=1)

    for lab in unique_labels:
        label_index = np.where(labels == lab)[0]
        if lab == -1:
            # Black used for noise.
            col = "black"
        elif len(label_index) == 1:
            col = DFLT_COL
        else:
            col = COLOR_LIST[lab]

        if family:
            for mark in range(min(pdb_prot_index), max(pdb_prot_index)+1):
                label_fam_index = np.intersect1d(label_index, np.where(np.asarray(family_map)==mark)[0])
                plt.plot(a[label_fam_index, 0],a[label_fam_index, 1],
                    "x" if lab == -1 else MARKER_LIST[pdb_prot_index[ind]], markerfacecolor=col,
                    markeredgecolor="k",)
        else:
            plt.plot(temp1[label_index],temp2[label_index], "x" if lab == -1 else "o", markerfacecolor=col, markeredgecolor="k",)
            
    plt.xlabel(r'sum(R%s) ($\AA$)'%str(c2))
    plt.ylabel(r'sum(R%s) ($\AA$)'%str(c1))
    if family:
        # plt.legend()
        legend_elements = [Line2D([], [], marker=mark, label=label, linestyle='None') for mark, label in zip(MARKER_LIST[:len(family)], family)]
        # ax.legend(handles=legend_elements)
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.04,1), loc="upper left")

def plot_stacked_histogram(r1: int, r2:int, dist_mats: list, res_lists: list, \
    labels: list, uniprot_seqs: list, color_texts: list = None, SMD:float=None):
    """ Plot the stacked histogram of the specified distance pair across the
    set of distance matrices colored on the plot based on their clustering. The
    title and residue labeling is colored based on the color_text objects.

    The default syntax is to accept lists of res_list, uniprot_seq, color_text.
    This allows the labeling of multiple proteins in the title.

    Parameters
    ----------
    r1 : int
        index mapping to a res_list
    r2 : int
        index mapping to a res_list
    dist_mats : list
        array of distance matrices (3D: PDB * res_list * res_list).
    res_lists : list
        2D list of residues corresponding one axis of the distance matrix
    inds_fc : list
        2D list of structure indices for each cluster (2D: cluster * structure indices)
    uniprot_seqs : list
        2D list of strings of one letter amino acid codes that reflect the sequence which
        the residue numbering is based off of
    color_texts : list
        list of color_text objects
    SMD : float
        Standardized mean difference value of the specified distance pair

    """
    min_val = 1000
    max_val = 0
    clust_value = []

    unique_labels = set(labels)
    for lab in unique_labels:
        label_index = np.where(labels == lab)[0]
        clust_value.append(dist_mats[label_index, r1, r2])
    flat = [v for l in clust_value for v in l]
    max_val = np.max(flat)
    min_val = np.min(flat)

    bins=np.arange(min_val-1,max_val+1,0.15)
   # print(bins)
    fig = plt.figure()
    n,bins,patches = plt.hist(x=clust_value, bins=bins, stacked=True, color=COLOR_LIST[:len(unique_labels)])
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    title = ""
    center = float(1)/float(len(res_lists)+1)
    for i, (res_list, uniprot_seq) in enumerate(zip(res_lists, uniprot_seqs)):
        res1 = uniprot_seq[res_list[r1]-1] + str(res_list[r1])
        res2 = uniprot_seq[res_list[r2]-1] + str(res_list[r2])
        text_center = center + center*i
        if color_texts:
            fig.text(text_center-0.05, 0.90, res1, ha="center", va="bottom", size="x-large",color=color_texts[i].return_color(res_list[r1]))
            fig.text(text_center, 0.90, "–", ha="center", va="bottom", size="x-large")
            fig.text(text_center+0.05,0.90, res2, ha="center", va="bottom", size="x-large",color=color_texts[i].return_color(res_list[r2]))
        else:
            fig.text(text_center-0.05, 0.90, res1, ha="center", va="bottom", size="x-large")
            fig.text(text_center, 0.90, "–", ha="center", va="bottom", size="x-large")
            fig.text(text_center+0.05,0.90, res2, ha="center", va="bottom", size="x-large")
    if SMD:
#         plt.legend("SMD: %.3f"%SMD)
        plt.annotate("SMD: %.3f"%SMD, xy=(0.97, 0.95), xycoords='axes fraction', size=14, ha='right', va='top', )

def run(dist_mats: np.ndarray, res_list: list, remove_missing: bool = True,
        replace_zeros: str = None, family: list = None, cumsum=0.8, hdbscan_args: dict = dict()):
    """ Pass in a list of the distance matrices and the corresponding residue
    lists and perform the hierarchical clustering and principal component
    analysis

    Parameters
    ----------
    dist_mats : np.ndarray
        array of distance matrices (3D: PDB * res_list * res_list).
    res_list : list
        list of residues
    k : int
        number of clusters
    remove_missing : bool
        bool whether or not to determine the subset of residues that are present
        in every distance matrix
    replace_zeros : str
        changes all of the zeros to the 'mean' or 'median' of that distance
    family : list
        if performing an analysis on a family of proteins, pass in a list of the
        protein names. if using this flag the dist_mats must be a 4D array
        (protein * PDB * res_list * res_list) and res_list will be a 2D array
        (protein * res_list). the list of proteins names must be the same length
        as the length of the first res_list axis and the first dist_mats axis
    hdbscan_args: dict
        dictionary of keywork arguments to pass to hdbscan

    Return
    ----------
    np.array
        projected coordinates following PCA
    list
        list of labels assigned by HDBSCAN
    np.array
        resulting matrices after removing zeros
    list
        list of residues
    list
        list of indices
    sklearn.decomposition.PCA
        PCA plot fit to features
    sklearn.cluster.HDBSCAN
        clustering method fit to datapoints
    """

    ## should the res_list be the same size as the length of the dist_mat

    if remove_missing and replace_zeros:
        raise ValueError('Cannot have flags remove_missing and replaces zeros at the same time')

    if family:
        for dist_mat_list in dist_mats:
            if np.array(dist_mat_list).shape[-1] != np.array(dist_mat_list).shape[-2]:
                raise ValueError('PCA calculates symetric distance matrices')
        if len(family) != len(dist_mats):
            raise ValueError('length of family list must equal length of the \
                              first axis of dist_mats')
        if len(family) != len(res_list):
            raise ValueError('length of family list must equal length of the \
                              first axis of res_list')
        index_map = [i for i,prot in enumerate(dist_mats) for _ in prot]

        if replace_zeros:
            dist_mats_1 = copy(dist_mats)
            for fam in family:
                print("replacing %s missing data with the %s distance"%(fam,replace_zeros))
                for i,prot_dist_mat in enumerate(dist_mats_1):
                    dist_mats[i] = replace_zeros_(prot_dist_mat, method=replace_zeros)
            ind_list = np.arange(0,len(res_list[0])) ## this will not work if the range of consecutive resiudes are not passed in
        ## flatten dist_mats to 3D
        dist_mats = np.array([dist_mat for dist_mat_list in dist_mats for dist_mat in dist_mat_list])
        # print(dist_mats.shape)
        if remove_missing:
            print("removing residues not available in every structure")
            dist_mats, _, ind_list = remove_missing_(dist_mats, res_list[0])
            print(ind_list)
            res_list = [prot_res[ind_list] for prot_res in res_list]
        else:
            ind_list = res_list
        # if replace_zeros:
        #      print("replacing missing data with the %s distance"%(replace_zeros))
        #      dist_mats = replace_zeros_(dist_mats, method=replace_zeros)
        #      ind_list = np.arange(0,len(res_list[0]))
        feats_list = triu_flatten(dist_mats, len(res_list[0]))
    else:
        if dist_mats.shape[-1] != dist_mats.shape[-2]:
            raise ValueError('PCA calculates symetric distance matrices')
        if remove_missing:
            print("removing residues not available in every structure")
            dist_mats, res_list, ind_list = remove_missing_(dist_mats, res_list)
        else:
            ind_list = res_list
        if replace_zeros:
            dist_mats = replace_zeros_(dist_mats, method=replace_zeros)
            ind_list = np.arange(0,len(res_list))
        index_map = None
        feats_list = triu_flatten(dist_mats, len(res_list))

    proj_coords, var_ratio, npy_pca = get_pca(feats_list, cumsum=cumsum)
    labels, hdb = pca_hdbscan(proj_coords, var_ratio, hdbscan_args, index_map, family)
    return(proj_coords, labels, dist_mats, res_list, ind_list, npy_pca, hdb)
