from collections import defaultdict
import sys


def default_value():
    return sys.maxint


class ClusterAggregator(object):
    def __init__(self):
        """
        A pair of forward and reverse dictionaries used to aggregate
        the cluster labels into global level labels.
        """
        self.fwd = defaultdict(default_value)
        self.rev = defaultdict(set)
        self.next_global_id = 0

    def __add__(self, other):
        """
        :param other: Another ClusterAggregator or a tuple containing
            an index and an iterable of partition level labels
        :rtype: ClusterAggregator
        :return: self
        Merges the clusters described by this ClusterAggregator with
        either another ClusterAggregator or an (index, iterable) tuple.
        """
        if type(other) == ClusterAggregator:
            # combine
            # I can't believe it was this simple
            for item in other.rev.iteritems():
                self + item
            pass
        else:
            # sequential
            index, pl_ids = other
            new_ids = set(pl_ids)
            # check if this point is noise
            first = next(iter(new_ids))
            if '-1' not in first and '*' not in first:
                global_id = self.next_global_id
                for new_id in new_ids:
                    # check if partition level label already recorded
                    if new_id in self.fwd:
                        global_id = min(global_id, self.fwd[new_id])
                if global_id == self.next_global_id:
                    # create new cluster
                    self.next_global_id += 1
                else:
                    # find overlapping clusters
                    overlaps = [self.fwd[new_id] for new_id in new_ids
                                if new_id in self.fwd]
                    # merge overlapping clusters
                    for gl_id in overlaps:
                        if gl_id != global_id:
                            # copy labels into new cluster
                            for pl_id in self.rev[gl_id]:
                                self[pl_id] = global_id
                            # delete older cluster
                            del self.rev[gl_id]
                # insert new cluster labels
                for new_id in new_ids:
                    self[new_id] = global_id
        return self

    def __setitem__(self, a, b):
        """
        :param a: Partition level cluster id
        :param b: Global level cluster id
        Set the value of fwd[a] to b and add a to the set in rev[b].
        """
        self.fwd[a] = b
        self.rev[b].add(a)
