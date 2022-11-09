import re
from anytree import RenderTree, NodeMixin, PreOrderIter


class TreeNode(NodeMixin):
    """Node class of a tree-like data structure to analyze the hierarchical
    clustering results.

    Parameters
    ----------
    name : str
        Name of node
    key : list
        List of strings that act as search key for the feature list in gen_children
    inds_fc : list
        2D list of structure indices for each cluster (2D: cluster * structure indices).
        inds_fc should only be passed in for the root node.
    num_cluster : int
        Total number of clusters. Should correspond to len(inds_fc)
    parent : TreeNode
        Reference to the parent node
    children : TreeNode
        References to the children node.
    other : str
        Name of sibling/child node other

    Attributes
    ----------
    cluster_count : list
        Root node attribute for size of each cluster
    repopulate : bool
        Flag indicating whether multiple siblings are names
    """

    def __init__(self, name: str, key: list = None, inds_fc: list = None,
                 num_cluster: int = 0, parent=None, children: list = None,
                 other: str = "Other"):
        if inds_fc:
            self.name = name
            self.parent = parent
            if children:
                self.children = children
            self.key = key
            self.cluster_count = [len(ind_fc) for ind_fc in inds_fc]
            self.inds_fc = inds_fc
            self.num_cluster = num_cluster
            self.repopulate = False
            self.other = other
        else:
            self.name = name
            self.cluster_count = [0 for i in range(num_cluster)]
            self.parent = parent
            self.inds_fc = [[] for i in range(num_cluster)]
            self.key = key
            self.num_cluster = num_cluster
            self.repopulate = False
            self.other = other
            if children:
                self.children = children

    def _update(self, cluster_ind: int, ind: int):
        """Helper function to populate the node. If repopulate flag is on then
        it checks if the index has already been populated

        Parameters
        ----------
        cluster_ind : int
            Description of parameter `cluster_ind`.
        ind : int
            Description of parameter `ind`.

        """
        if self.repopulate:
            if ind in self.inds_fc[cluster_ind]:
                return
            self.cluster_count[cluster_ind] += 1
            self.inds_fc[cluster_ind].append(ind)
        else:
            self.cluster_count[cluster_ind] += 1
            self.inds_fc[cluster_ind].append(ind)

    def _refresh_other(self):
        """If you try to gen_children on a node that already has children, then
        this helper function is called to move the indices in the 'other' node
        to populate other nodes

        Returns
        -------
        TreeNode
            'Other' node

        """
        children_names = [node.name for node in self.children]
        not_other_ind = [i for i, name in enumerate(
            children_names) if name not in [self.other, "None"]]
        node_list = [list(self.children)[x] for x in not_other_ind]
#         print(node_list)
        not_other_inds_fc = list(set([y for node in node_list for x in node.inds_fc for y in x]))
        ind1 = children_names.index(self.other)
        cb = [node for node in self.children][ind1]
        for i, ind_fc in enumerate(cb.inds_fc):
            del_list = []
            for j, ind in enumerate(ind_fc):
                if ind in not_other_inds_fc:
                    del_list.append(j)
            for j in del_list[::-1]:
                del ind_fc[j]
        for i, inds in enumerate(cb.inds_fc):
            cb.cluster_count[i] = len(inds)
        return cb

    def _key_check(self, node, feat: str, not_key: list = None, modif: bool = False):
        """Helper function for gen_children to check the feature for the key
        as specified by the node

        Parameters
        ----------
        node : TreeNode
            Node containing the key
        feat : str
            feature string
        not_key : list
            list of keywords to exclude
        modif : bool
            flag to specifically search for modifications

        Returns
        -------
        list
            list of found matches to the key

        """
        match = []
        if modif:
            if feat[3].lower() == ca.key:
                match = [feat[3]]
        else:
            feat_spl = re.split(r'\s|-|[.,:;]', str(feat).lower())
            if isinstance(node.key, list):
                for k in node.key:
                    if k in feat_spl:
                        if not_key:
                            if not any([x in feat_spl for x in not_key]):
                                match = [x for x in feat_spl if x == k]
                        else:
                            match = [x for x in feat_spl if x == k]
            else:
                if node.key in feat_spl:
                    if not_key:
                        if not any([x in feat_spl for x in not_key]):
                            match = [x for x in feat_spl if x == node.key]
                    else:
                        match = [x for x in feat_spl if x == node.key]
        return match

    def gen_children(self, name: str, key: list, feature_list: list, modif: bool
                     = False, not_key: list = None, other: str = "Other"):
        """Generates children nodes (name, other, none). If the feature matches
        key populates child node name with the index. If the feature does not
        match then populate the child node other. If the feature has no value
        then populate the child node none.

        Can provide a not_key to exclude certain strings.

        Parameters
        ----------
        name : str
            Name of child node
        key : list
            list of keywords to search for
        feature_list : list of lists
            list of features
        modif : bool
            flag to specifically search for modifications
        not_key : list
            list of keywords to exclude
        other : str
            Name of other node

        Returns
        -------
        tuple
            Returns tuple of TreeNodes representing name, other and none

        """
        ca = TreeNode(name=name, key=key, num_cluster=self.num_cluster, other=other, parent=self)
        children_names = [node.name for node in self.children]
        refresh = False
        if self.other in children_names and "None" in children_names:
            ind1 = children_names.index(self.other)
            cb = list(self.children)[ind1]
            cb.repopulate = True
            ind2 = children_names.index("None")
            cc = list(self.children)[ind2]
            cc.repopulate = True
            refresh = True

            order = list(range(0, len(children_names)))
            order[-2] = ind1
            order[-1] = ind2
            order[-3] = len(children_names)-1
            self.children = [self.children[i] for i in order]
        else:
            cb = TreeNode(self.other, key=self.other,
                          num_cluster=self.num_cluster, parent=self, other=other)
            cc = TreeNode("None", num_cluster=self.num_cluster, parent=self)

        if not self.inds_fc:
            raise ValueError("Unable to generate children, node should have\
                              inds_fc value")

        for i, ind_fc in enumerate(self.inds_fc):
            for ind in ind_fc:
                feat_l = feature_list[ind]
                if feat_l:
                    found = False
                    for feat in feat_l:
                        match = self._key_check(ca, feat, not_key)
#                             print(feat, feat_spl, not_key, match)
                        if match and not found:
                            ca._update(i, ind)
                            found = True
                            continue
                    if not found:
                        cb._update(i, ind)
                else:
                    cc._update(i, ind)
        if refresh:
            cb = self._refresh_other()
        return(ca, cb, cc)

    def search_leaves_for_ind(self, ind: int):
        """Traverses through leaves to search for a particular index. Prints out
        string describing the traversal required to find the leaf node

        Parameters
        ----------
        ind : int
            Index to search for

        Returns
        -------
        TreeNode
            leaf node containing the index

        """
        for ch in PreOrderIter(self):
            if not ch.children:
                # print(ch.inds_fc)
                for i, ind_fc in enumerate(ch.inds_fc):
                    if ind in ind_fc:
                        trav = ch
                        text = ch.name+"(%i)" % i
                        while (trav.parent):
                            text = trav.parent.name + "->" + text
                            trav = trav.parent
                        print(text)
                        return ch

    def delete_sibling_node(self, name: str):
        """Iterates through sibling nodes and removes by name

        Parameters
        ----------
        name : str
            Name of node to delete

        """
        for node in self.parent.children:
            if node.name == name:
                node.parent = None


def print_tree(root: TreeNode):
    """Traverses through nodes and prints the tree in a nice format

    Parameters
    ----------
    root : TreeNode
        Root node of tree

    """
    for pre, _, node in RenderTree(root):
        # print(pre)
        treestr = "%s%s %s %i" % (pre, node.name, node.cluster_count, sum(node.cluster_count))
        print(treestr)
