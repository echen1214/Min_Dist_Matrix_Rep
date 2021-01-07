from collections import Counter
from sklearn import decomposition
from colored import fg, bg, attr
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt



""" TODO
- x get list of residues that are present in all structures
- organize data properly
- x calculate principal compenents
    - x output PCA plot/ eigenvalue contributions
    - create interactive plot where you can interactively select ranges in jupyter
      notebook and it'll return the PDB
- x hierarchical clustering
    - x output hierarchical clustering plot
    - create interactive plot where you can interactively select ranges in jupyter
      notebook and it'll return the PDB
- x determine the important distances - Z-score
    - create interactive plot where you can interactively select ranges in jupyter
      notebook and it'll return the PDB
- update color_text to pass in regions and color list


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

COLOR_LIST = ['g','r','c','m','y','k']
DFLT_COL = "#808080"

def remove_missing_(dist_mats: np.ndarray, res_list: list):
    """ Pass in a list of the distance matrices and the corresponding residue lists,
    determine the subset of residues that are present in every distance matrix

    Parameters
    ----------
    dist_mats : np.ndarray
        array of distance matrices (3D: PDB * res_list * res_list).
    res_list : list
        list of residues corresponding to rows and columns of the distance matrix

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
    """ Maybe change the function name """
    missing_res = Counter()

    for mat in dist_mats:
        missing = np.where(~mat.any(axis=1))[0]
        missing_res.update(missing)

    #print(missing_res.keys())
    ind_cons = [x for x in range(len(res_list)) if x not in missing_res]

    res_cons= [res_list[x] for x in range(len(res_list)) if x not in missing_res]

    print(len(ind_cons), res_cons)

    dist_mats_cons = []
    for mat in dist_mats:
        dist_mats_cons.append(mat[np.ix_(ind_cons, ind_cons)])
    return(np.array(dist_mats_cons), res_cons, ind_cons)

def triu_flatten(dist_mats: np.ndarray, res_list: list):
    """ Return an array of flattened 1D upper triangular values (features) of each
    residue-residue distance matrix. These values include all those above the
    diagonal of the matrix (k=1)

    Parameters
    ----------
    dist_mats : np.ndarray
        array of distance matrices (3D: PDB * res_list * res_list | 2D: res_list * res_list)
    res_list : list
        list of residues corresponding to rows and columns of the distance matrix

    Returns
    -------
    np.ndarray
        array of 1D flattened distance matrices (2D: PDB * features | 1D: res_list)

    """
    triu_ind = np.triu_indices(len(res_list), k=1)
    feats_list = []
    print(len(dist_mats.shape))
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
    print("med_ind", med_ind)
    sort_ind = np.argsort(d_mat[med_ind])
    return ([ind_map[x] for x in sort_ind])


def clustering(feats: np.ndarray, k: int, method: str = 'ward', criterion: str = 'maxclust'):
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
    method : str
        method options, https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    criterion : str
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
        # pdb_list_fc = [pdb_list[x] for x in ind_fc]
        # chain_list_fc = [chain_list[x] for x in ind_fc]

        # pdb_clust_temp = [pdb_list_fc[ind] for ind in medoid_ind]
        # chain_clust = [chain_list_fc[ind] for ind in medoid_ind]
        #pdb_clust.append(pdb_clust_temp)

        inds_fc.append(ind_fc)
        medoid_ind_list.append(medoid_ind)
    print("link_cols")
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
    print("dendrogram")
    plt.figure(figsize=(25, 5))
    R = sch.dendrogram(Z, no_plot=False,leaf_rotation=90., color_threshold=None,\
                        link_color_func=lambda x: link_cols[x], no_labels=True)
    return (inds_fc, org_color_dict, medoid_ind_list)

def plot_analy(npy_pca: decomposition.PCA, feats: np.ndarray, k: int):
    """Plots the dendrogram and the PC1/PC2 plot
    Parameters
    ----------
    npy_pca : sklearn.decomposition.PCA
        PCA object
    feats_list : np.ndarray
        array of 1D flattened distance matrices
    k : int
        number of clusters

    """
    inds_fc, org_color_list, medoid_ind_list = clustering(feats, k)
    plot_pca(npy_pca, feats, inds_fc)
    return(inds_fc, medoid_ind_list)

def plot_pca(npy_pca: decomposition.PCA, feats: np.ndarray, inds_fc: list):
    """Centers and plots the PC1/PC2 plots and prints out the percentage of variance
    explained by PC1/PC2

    The cluster colors are green, red, cyan, maroon, yellow, black. Any unclusted
    structures are colored a default color, grey.

    Parameters
    ----------
    npy_pca : type
        PCA object
    feats : np.ndarray
        array of 1D flattened distance matrices
    inds_fc : list
        2D list of structure indices for each cluster (2D: cluster * structure indices)

    """
    print("plot PCA")
    feats -= np.mean(feats, axis=0)
    a = np.dot(feats, npy_pca.components_.T)
    plt.figure()
    ax = plt.axes()
    index = 0
    for i,ind_fc in enumerate(inds_fc):
        cluster = a[ind_fc]
        if len(ind_fc)==1:
            color=DFLT_COL
        else:
            color = COLOR_LIST[index]
            index=index+1
        print("cluster size:", len(cluster), color)
        ax.scatter(cluster[:,0], cluster[:,1], color=color)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    print(npy_pca.explained_variance_ratio_[:2])

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

def calc_cluster_zscore(cluster1_inds: list, cluster2_inds: list, feats: np.ndarray):
    """Calculate the Z-score of the pairwise residue-residue distances (features)
    between two clusters.

    Z-score = average(cluster 1 features) - average(cluster 2 features) / (sqrt (std(cluster 1 features) * std(cluster 2 features)))
    This value is then biased by dividing by (minimum feature value - 1.5)

    Parameters
    ----------
    cluster1_inds : list
        list of indices of the feats array that reflect cluster 1
    cluster2_inds : list
        list of indices of the feats array that reflect cluster 2
    feats : np.ndarray
        array of 1D flattened distance matrices

    Returns
    -------
    tuple
        the biased z-score value, minimum distance list

    """
    ''' include different normalization functions '''
    mean_feats_c1 = np.average(feats[cluster1_inds], axis=0)
    mean_feats_c2 = np.average(feats[cluster2_inds], axis=0)

    std_feats_c1 = np.std(feats[cluster1_inds], axis=0)
    std_feats_c2 = np.std(feats[cluster2_inds], axis=0)

    zscore_diff_c1c2 = np.divide(np.subtract(mean_feats_c1, mean_feats_c2), \
                                 np.sqrt(np.multiply(std_feats_c1, std_feats_c2)))
    zscore_diff_c1c2_idx = np.argsort(zscore_diff_c1c2)[::-1]

    concat_ind = cluster1_inds + cluster2_inds
    min_dist = np.amin([feats[x] for x in concat_ind], axis=0)
    zscore_new = np.divide(zscore_diff_c1c2, (np.subtract(min_dist,1.5))) #np.divide(np.log(min_dist), np.log(5)))
    return (zscore_new, min_dist)

def plot_zscore_distrib(cluster1: int, cluster2: int, feats: np.ndarray, \
                        zscore: np.ndarray, xcutoff: int=3.5, ycutoff: int=5):
    """Plots the distribution of Z-scores vs minimum distance. Each data point
    represented one distance pair

    Parameters
    ----------
    cluster1 : int
        cluster index number
    cluster2 : int
        cluster index number
    feats : np.ndarray
        array of 1D flattened distance matrices
    zscore : np.ndarray
        array of Z-score values for each distance pair
    xcutoff : int
        distance cutoff value
    ycutoff : int
        Z-score cutoff value

    """
    min_feats = np.amin(feats, axis=0)
    # min_feats_idx = np.argsort(min_feats)
    min_zscore = np.stack((min_feats,zscore),axis=1)
    plt.figure()
    plt.scatter(min_zscore[:,0],min_zscore[:,1])
    plt.xlabel("minimum distance along distance pair between the two clusters")
    plt.ylabel("Z-score difference between %i and %i/(min_dist-1.5)"%(cluster1+1, cluster2+1))
    plt.axvline(x=xcutoff, color="red")
    plt.axhline(y=0, color='black')
    plt.axhline(y=ycutoff, color='red', linestyle='dashed')
    plt.axhline(y=-ycutoff, color='red', linestyle='dashed')
    if xcutoff > 10:
        plt.xlim(-0.5,xcutoff)
    else:
        plt.xlim(-0.5, 10)

###, color_list, region_list):
def color_text(res: str, res_num: int):
    """Creates colored string corresponding to the specified regions and region
    colors

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
    color_list  = ['red','orange_red_1', 'orange_1', 'yellow', 'green', 'cyan', 'blue', 'purple_1b', 'magenta']
    region_list = [np.arange(12,17), np.arange(45,58),[33,51,127,145],np.arange(80,87),\
                np.arange(125,132),np.arange(145,172),np.arange(182,197)]
    if res_num in region_list[2]:
        return(fg("%s"%color_list[2])+res+str(res_num)+attr(0))
    for i,reg in enumerate(region_list):
        if res_num in reg:
            return(fg("%s"%color_list[i])+res+str(res_num)+attr(0))
    return res+str(res_num)

def plot_zscore(cluster1, cluster2, feats, min_dist, zscore, res_list, uniprot_seq, top=10, \
                xcutoff=3.5, ycutoff=5):
    zscore_new_idx = np.argsort(np.abs(zscore))[::-1]
    plot_zscore_distrib(cluster1, cluster2, feats, zscore, \
                        xcutoff=xcutoff, ycutoff=ycutoff)

    top_ind = 0
    feat_idx = []
    for idx in zscore_new_idx:
        if min_dist[idx] < xcutoff and abs(zscore[idx]) > ycutoff and top_ind < top:
            top_ind+=1
            x,y=triu_getXY(idx,m=len(res_list),k=1)
#             plot_stacked_histogram(x,y,mats_list,[res_list],inds_fc,var=None)
            feat_idx.append((x,y,idx,zscore[idx]))
            r1 = uniprot_seq[res_list[x]-1]
            r1_id = res_list[x]
            r2 = uniprot_seq[res_list[y]-1]
            r2_id = res_list[y]
            print("%s-%s: %.3f, %.3f\n"%(color_text(r1,r1_id), color_text(r2,r2_id),min_dist[idx],zscore[idx]))
            if np.isinf(zscore[idx]):
                print("Warning: infinite value may result from std = 0")
                top_ind -= 1
    print(feat_idx)

def run(dist_mats: np.ndarray, res_list: list, k: int, remove_missing: bool = True):
    """ Pass in a list of the distance matrices and the corresponding residue
    lists and perform the hierarchical clustering and principcal component
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

    """
    if dist_mats.shape[1] != dist_mats.shape[2]:
        raise ValueError('PCA calculates symetric distance matrices')

    if remove_missing:
        dist_mats, res_list, ind_list = remove_missing_(dist_mats, res_list)

    feats_list = triu_flatten(dist_mats, res_list)
    print("PCA")
    npy_pca = decomposition.PCA()
    npy_pca.fit(feats_list)

    inds_fc, medoid_ind_list = plot_analy(npy_pca, feats_list, k)
    return (dist_mats, res_list, ind_list, inds_fc, medoid_ind_list)
    # return (npy_pca)
