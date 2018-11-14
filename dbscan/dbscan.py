import pyspark as ps
import sklearn.cluster as skc
from scipy.spatial.distance import *
from partition import KDPartitioner
from aggregator import ClusterAggregator
from operator import add
import numpy as np

LOGGING = False


def dbscan_partition(iterable, params):
    """
    :type iterable: iter
    :param iterable: iterator yielding ((key, partition), vector)
    :type params: dict
    :param params: dictionary containing sklearn DBSCAN parameters
    :rtype: iter
    :return: ((key, cluster_id), v)
    Performs a DBSCAN on a given partition of the data
    """
    # read iterable into local memory
    data = list(iterable)
    (key, part), vector = data[0]
    x = np.array([v for (_, __), v in data])
    y = np.array([k for (k, _), __ in data])
    # perform DBSCAN
    model = skc.DBSCAN(**params)
    c = model.fit_predict(x)
    cores = set(model.core_sample_indices_)
    # yield (key, cluster_id), non-core samples labeled with *
    for i in xrange(len(c)):
        flag = '' if i in cores else '*'
        yield (y[i], '%i:%i%s' % (part, c[i], flag))


def map_cluster_id((key, cluster_id), broadcast_dict):
    """
    :type broadcast_dict: pyspark.Broadcast
    :param broadcast_dict: Broadcast variable containing a dictionary
        of cluster id mappings
    :rtype: int, int
    :return: key, cluster label
    Modifies the item key to include the remapped cluster label,
    choosing the first id if there are multiple ids
    """
    cluster_id = next(iter(cluster_id)).strip('*')
    cluster_dict = broadcast_dict.value
    if '-1' not in cluster_id and cluster_id in cluster_dict:
        return key, cluster_dict[cluster_id]
    else:
        return key, -1


class DBSCAN(object):
    """
    :eps: nearest neighbor radius
    :min_samples: minimum number of sample within radius eps
    :metric: distance metric
    :max_partitions: maximum number of partitions used by KDPartitioner
    :data: copy of the data used to train the model including
    :result: RDD containing the (key, cluster label) pairs
    :bounding_boxes: dictionary of BoundingBoxes used to partition the
        data
    :expanded_boxes: dictionary of BoundingBoxes expanded by 2 eps in
        all directions, used to partition data
    :neighbors: dictionary of RDD containing the ((key, cluster label),
        vector) for data within each partition
    :cluster_dict: dictionary of mappings for neighborhood cluster ids
        to global cluster ids
    """

    def __init__(self, eps=0.5, min_samples=5, metric=euclidean,
                 max_partitions=None):
        """
        :type eps: float
        :param eps: nearest neighbor radius
        :type min_samples: int
        :param min_samples: minimum number of samples within radius eps
        :type metric: callable
        :param metric: distance metric (should be
            scipy.spatial.distance.euclidian or
            scipy.spatial.distance.cityblock)
        :type max_partitions: int
        :param max_partitions: maximum number of partitions in
            KDPartitioner
        Using a metric other than euclidian or cityblock/Manhattan may
        not work as the bounding boxes expand in such a way that
        other metrics may return distances less than eps for points
        outside the box.
        """
        self.eps = eps
        self.min_samples = int(min_samples)
        self.metric = metric
        self.max_partitions = max_partitions
        self.data = None
        self.result = None
        self.bounding_boxes = None
        self.expanded_boxes = None
        self.neighbors = None
        self.cluster_dict = None

    def train(self, data):
        """
        :type data: pyspark.RDD
        :param data: (key, k-dim vector like)
        Train the model using a (key, vector) RDD
        """
        parts = KDPartitioner(data, self.max_partitions)
        self.data = data
        self.bounding_boxes = parts.bounding_boxes
        self.expanded_boxes = {}
        self._create_neighborhoods()
        # repartition data set on the partition label
        self.data = self.data.map(lambda ((k, p), v): (p, (k, v))) \
            .partitionBy(len(parts.partitions)) \
            .map(lambda (p, (k, v)): ((k, p), v))
        # create parameters for sklearn DBSCAN
        params = {'eps': self.eps, 'min_samples': self.min_samples,
                  'metric': self.metric}
        # perform dbscan on each part
        self.data = self.data.mapPartitions(
            lambda iterable: dbscan_partition(iterable, params))
        self.data.cache()
        self._remap_cluster_ids()

    def assignments(self):
        """
        :rtype: list
        :return: list of (key, cluster_id)
        Retrieve the results of the DBSCAN
        """
        return self.result.collect()

    def _create_neighborhoods(self):
        """
        Expands bounding boxes by 2 * eps and creates neighborhoods of
        items within those boxes with partition ids in key.
        """
        neighbors = {}
        new_data = self.data.context.emptyRDD()
        for label, box in self.bounding_boxes.iteritems():
            expanded_box = box.expand(2 * self.eps)
            self.expanded_boxes[label] = expanded_box
            neighbors[label] = self.data.filter(
                lambda (k, v): expanded_box.contains(v)) \
                .map(lambda (k, v): ((k, label), v))
            new_data = new_data.union(neighbors[label])
        self.neighbors = neighbors
        self.data = new_data

    def _remap_cluster_ids(self):
        """
        Scans through the data for collisions in cluster ids, creating
        a mapping from partition level clusters to global clusters.
        """
        labeled_points = self.data.groupByKey()
        labeled_points.cache()
        mapper = labeled_points.aggregate(ClusterAggregator(), add, add)
        b_mapper = self.data.context.broadcast(mapper.fwd)
        self.result = labeled_points \
            .map(lambda x: map_cluster_id(x, b_mapper)) \
            .sortByKey()
        self.result.cache()
