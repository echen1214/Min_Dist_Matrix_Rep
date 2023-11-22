import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def get_var_mat(mats, norm: int = 0):
    """_summary_

    Parameters
    ----------
    mats : np.array
        _description_
    norm : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    var = np.zeros((len(mats[0]), len(mats[0][0])))
    # print(len(mats))
    # print (var.shape)

    # can maybe make this more elegeant using axis = 0)
    # square rooting these values actually result in the standard deviation
    for i, j in np.ndindex(var.shape):
        # print(i,j)
        value = []
        for mat in mats:
            if i != j and (mat[i][j] == 0):
                continue
            value.append(mat[i][j])
        if norm == 0:
            var[i, j] = np.sqrt(np.var(value))
        elif norm == 1:
            if np.average(value) == 0:
                continue  # skip diagonal
            else:
                var[i, j] = np.sqrt(np.var(value))/np.average(value)
        elif norm == 2:
            if np.average(value) == 0:
                continue
            else:
                var[i, j] = np.sqrt(np.var(value))/np.min(value)
        elif norm == 3:
            if np.average(value) == 0:
                continue
            else:
                var[i, j] = (np.max(value)-np.min(value))/np.min(value)
        else:
            print("norm value must be 0,1,2,3")
            break
        # print(value)
    return var


def triag_num(n):
    return int(n*(n+1)/2)


def get_residue_pairs_below(var1, binx):
    res_pair = []  # variance for xy above diagonal
    for (x, y), vari in np.ndenumerate(var1):
        if x > y:
            if vari < binx:
                res_pair.append((x, y))
    return (res_pair)


def find_max_len_clique(list_clique):
    max_len_clique = 0
    clique_ind = []
    for i, clique in enumerate(list_clique):
        if len(clique) > max_len_clique:
            clique_ind = []
            max_len_clique = len(clique)
            clique_ind.append(i)
        elif len(clique) == max_len_clique:
            clique_ind.append(i)

    max_clique = [np.sort(list_clique[i]) for i in clique_ind]
    return (max_len_clique, max_clique)


def plot_number_of_max_clique(ax, bin_edges, var_clique_list):
    ax2 = ax.twinx()
    ax2.plot(bin_edges, [len(clique_list)
             for clique_list in var_clique_list], color="red")
    ax2.set_ylabel("Number of max cliques", color="red")
    plt.axhline(y=1, color="red", linestyle="--")


def get_graph(var1, cutoff, ident):
    G = nx.Graph()
    # bin1 = get_bins(var1, cutoff, bin_step, max_var)
    # print("cutoff %f"%bin1)
    excl_res_pair = get_residue_pairs_below(var1, cutoff)
    # print(len(excl_res_pair))
    for x, y in excl_res_pair:
        # print(ident[x],ident[y],var1[x,y])
        G.add_edge(ident[x], ident[y], weight=var1[x, y])
    return (G)


def plot_clique_range(var, ident, max_var, bin_step):
    freq, freq2 = [], []
    max_clique_list,  max_clique_list2 = [], []
    length = int(max_var/bin_step)
    # print(length)

    tstep = 10
    for i in range(length):
        # print("get graph")
        if i % tstep == 0:
            print(i, length)
        threshold = i*bin_step
        max_len_temp, max_clique_temp = get_clique_residues_at_threshold(
            var, threshold, ident)
        freq.append(max_len_temp)
        max_clique_list.append(max_clique_temp)

        ## to plot the second largest unique clique -- for hinge like proteins     ##
        ## exclude largest clique from var and ident -> find clique with remaining ##
        # if len(max_clique_temp)==1:
        #    #print(max_clique_temp)
        #    max_clique_temp_all = [j for i in max_clique_temp for j in i]
        #    second_ident = [i for i,x in enumerate(ident) if x not in max_clique_temp_all]
        #    var2 = var[np.ix_(second_ident, second_ident)]
        #
        #    bin2_temp = get_graph(var2, i, np.arange(0,len(second_ident)), bin_step, max_var)
        #    list_clique_temp = list(nx.find_cliques(bin2_temp))
        #    max_len_temp, max_clique_temp = find_max_len_clique(list_clique_temp)
        #    freq2.append(max_len_temp)
        #    max_clique_list2.append(max_clique_temp)
        # else: freq2.append(None)
    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, max_var, bin_step), freq, label="Largest clique")
    # ax.scatter(np.arange(0,max_var,bin_step), freq2, color = 'green', label = "Second largest unique clique")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Maximum clique size", color='#1f77b4')
    ax.set_ylim(top=len(ident))
    fig.legend()
    # print("plot number of max clique")
    plot_number_of_max_clique(ax, np.arange(
        0, max_var, bin_step), max_clique_list)
    # plot_number_of_max_clique(ax,np.arange(0,max_var,bin_step),max_clique_list2, "green", )
    # , max_clique_list2)
    return (np.arange(0, max_var, bin_step), max_clique_list)


def get_graph_1(var1, cutoff, ident):
    G = nx.Graph()
    # bin1 = get_bins(var1, cutoff, bin_step, max_var)
    # print("cutoff %f"%bin1)
    excl_res_pair = get_residue_pairs_below(var1, cutoff)
    # print(len(excl_res_pair))
    for x, y in excl_res_pair:
        # print(ident[x],ident[y],var1[x,y])
        G.add_edge(ident[x], ident[y], weight=var1[x, y])
    return (G)


def get_overlapping_clique_residues(clique1, clique2):
    overlap = set(clique1).intersection(set(clique2))
    return (sorted(list(overlap)))
# c1 = [107, 108, 109, 110, 111, 112, 113, 188, 191, 261, 262, 263, 264, 265, 266, 267, 275, 276, 277, 283, 285, 286]
# c2 = [103, 104, 105, 106, 107, 108, 110, 112, 114, 116, 117, 141, 191, 276, 277, 279, 280, 286]
# print(get_overlapping_clique_residues(c1,c2))


def get_overlapping_clique_residues(clique1, clique2):
    overlap = set(clique1).intersection(set(clique2))
    return (sorted(list(overlap)))


def get_clique_residues_at_threshold(var_list, threshold, ident):
    graph = get_graph_1(var_list, threshold, ident)
    list_clique_temp = list(nx.find_cliques(graph))
    max_len_temp, max_clique_temp = find_max_len_clique(list_clique_temp)
    return (max_len_temp, max_clique_temp)


def get_superpos_res(var_list_1, var_list_2, threshold, ident):
    len1, clique1 = get_clique_residues_at_threshold(
        var_list_1, threshold, ident)
    len2, clique2 = get_clique_residues_at_threshold(
        var_list_2, threshold, ident)

    clique1 = (set(clique1[0]).intersection(*clique1[1:]))
    clique2 = (set(clique2[0]).intersection(*clique2[1:]))
    # print(clique1,clique2)
    # print(type(clique1), type(clique2))
    return (get_overlapping_clique_residues(list(clique1), list(clique2)))
