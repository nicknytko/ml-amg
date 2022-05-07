import numpy as np


class DisjointSets:
    '''
    An implementation of a disjoint-set data structure.
    A set of N elements are distributed into disjoint, non-overlapping sets; initialized
    into N singleton sets.  Elements from different sets can then be unioned together to
    combine sets.
    '''

    def __init__(self, N):
        self.sets = np.ones(N, dtype=np.int64) * -1

    def get_root(self, i):
        '''
        Get the "root element" of a set.
        Returns an arbitrary (but deterministic) element that is the representative root
        element for the set.  Two elements are in the same set if they have the same root.

        Parameters
        ----------
        i : integer
          Element to find root of.

        Returns
        -------
        root : integer
          Index of root element.
        '''

        if self.sets[i] == -1:
            return i
        else:
            root = self.get_root(self.sets[i])
            self.sets[i] = root
            return root

    def union(self, i, j):
        '''
        Combine the sets of two elements together.

        Parameters
        ----------
        i : integer
          Index of the first element
        j : integer
          Index of the second element

        Returns
        -------
        performed_union : boolean
          True if the two elements were in different unions and the sets were combined.
          False if the elements were already in the same set.
        '''

        parent = min(i,j)
        unionee = max(i,j)
        parent_root = self.get_root(parent)
        unionee_root = self.get_root(unionee)

        if not self.are_connected(parent, unionee):
            self.sets[unionee_root] = parent_root
            return True
        else:
            return False

    def are_connected(self, i, j):
        '''
        Checks if two elements are connected (in the same set)

        Parameters
        ----------
        i : integer
          Index of the first element
        j : integer
          Index of the second element

        Returns
        -------
        connected : boolean
          True if the two elements are in the same set.
        '''

        return self.get_root(i) == self.get_root(j)

    def get_num_disjoint(self):
        '''
        Counts the number of disjoint sets.
        Note this is an O(n) operation.  If you are iteratively unioning elements, it may
        be faster to manually keep count of sets as you union things together.

        Returns
        -------
        count : integer
          Number of disjoint sets that are present.
        '''

        return np.sum(self.sets == -1)

    def get_disjoint_sets(self):
        '''
        Returns all of the sets as a list of lists.  Each inner list contains
        elements that are in the same disjoint set.

        Returns
        -------
        sets : list of lists
          List of each disjoint set.
        '''

        roots = {}

        # Create set for each root
        for i, e in enumerate(self.sets):
            if e == -1:
                roots[i] = [i]

        # Now populate each set with its child elements
        for i, e in enumerate(self.sets):
            if e != -1:
                roots[self.get_root(e)].append(i)

        return list(roots.values())
