from Queue import Queue
from geometry import BoundingBox
from operator import add
import numpy as np
import pyspark as ps


def median_search_split(partition, axis, next_part):
    """
    :type partition: pyspark.RDD
    :param partition: pyspark RDD ((key, partition label) , k-dim
        vector like)
    :type axis: int
    :param axis: axis to split on
    :type next_part: int
    :param next_part: next partition label
    :return: part1, part2, median: part1 and part2 are RDDs with the
        same structure as partition, where the split was made
    :rtype: pyspark.RDD, pyspark.RDD, float
    Split the given partition into equal sized partitions along the
    given axis.
    """
    sorted_values = partition.map(lambda ((k, p), v): v[axis]).sortBy(
        lambda v: v).collect()
    median = sorted_values[
        len(sorted_values) / 2]  # need a better way to find the median
    part1 = partition.filter(lambda ((k, p), v): v[axis] < median)
    part2 = partition.filter(lambda ((k, p), v): v[axis] >= median).map(
        lambda ((k, p), v): ((k, next_part), v))
    return part1, part2, median


def mean_var_split(partition, k, axis, next_label, mean, variance):
    """
    :type partition: pyspark.RDD
    :param partition: pyspark RDD ((key, partition label), k-dim vector
        like)
    :type k: int
    :param k: number of dimension in vector data
    :type axis: int
    :param axis: axis to perform the split on
    :type next_label: int
    :param next_label: next partition label
    :type mean: float
    :param mean: mean of the given partition along the given axis
    :type variance: float
    :param variance: variance of the given partition along the given
        axis
    :return: part1, part2, median: part1 and part2 are RDDs with the
        same structure as partition, where the split was made
    :rtype: pyspark.RDD, pyspark.RDD, float
    Search for the median using the mean and variance and split into
    approximately equal sized partitions.
    Checks for boundaries that split the data into the most equal size
    partitions where the boundaries are at mean + [-0,9, -0.6, -0.3, 0,
    0.9, 0.6, 0.3] * std dev
    """
    std_dev = np.sqrt(variance)
    bounds = np.array([mean + (i - 3) * 0.3 * std_dev for i in xrange(7)])
    counts = partition.aggregate(np.zeros(7),
                                 lambda x, (_, v):
                                 x + 2 * (v[axis] < bounds) - 1,
                                 add)
    counts = np.abs(counts)
    boundary = bounds[np.argmin(counts)]
    part1 = partition.filter(lambda (_, v): v[axis] < boundary)
    part2 = partition.filter(lambda (_, v): v[axis] >= boundary).map(
        lambda ((key, _), v): ((key, next_label), v))
    return part1, part2, boundary


def min_var_split(partition, k, next_label):
    """
    :type partition: pyspark.RDD
    :param partition: pyspark RDD ((key, partition label), k-dim vector
        like)
    :type k: int
    :param k: dimensionality of the vectors in partition
    :type next_label: int
    :param next_label: next partition label
    :rtype: (pyspark.RDD, pyspark.RDD, float), int
    :return: (part1, part2, median), axis
    Split the given partition into equal sized partitions along the
    axis with greatest variance.
    """
    moments = partition.aggregate(np.zeros((3, k)),
                                  lambda x, (keys, vector): x + np.array(
                                      [np.ones(k), vector, vector ** 2]),
                                  add)
    means = moments[1] / moments[0]
    variances = moments[2] / moments[0] - means ** 2
    axis = np.argmax(variances)
    return mean_var_split(partition, k, axis, next_label, means[axis],
                          variances[axis]), axis
    # return median_search_split(partition, axis, next_label), axis


class KDPartitioner(object):
    """
    :partitions: dictionary of int => RDD containing the initial data
        filtered by the corresponding BoundingBox
    :bounding_boxes: dictionary of int => BoundingBox for that
        partition label
    :result: union of the RDD in partitions
    :k: dimensionality of the data
    :max_partitions: maximum number of partitions
    :split_method: string representing the method for splitting
        partitions
    """

    def __init__(self, data, max_partitions=None, k=None,
                 split_method='min_var'):
        """
        :type data: pyspark.RDD
        :param data: pyspark RDD (key, k-dim vector like)
        :type max_partitions: int
        :param max_partitions: maximum number of partition to split
            into
        :type k: int
        :param k: dimensionality of the data
        :type split_method: str
        :param split_method: method for splitting on axis - 'min_var'
            minimizes the variance in each partition, 'rotation'
            cycles through the axis
        Split a given data set into approximately equal sized partition
        (if max_partitions is a power of 2 ** k) using binary tree
        methods
        """
        self.split_method = split_method \
            if split_method in ['min_var', 'rotation'] else 'min_var'
        self.k = int(k) if k is not None else len(data.first()[1])
        self.max_partitions = int(
            max_partitions) if max_partitions is not None else 4 ** self.k
        data.cache()
        box = data.aggregate(BoundingBox(k=self.k),
                             lambda total, (_, v): total.union(BoundingBox(v)),
                             lambda total, v: total.union(v))
        first_partition = data.map(lambda (key, value): ((key, 0), value))
        self._create_partitions(first_partition, box)
        self.result = data.context.emptyRDD()
        for partition in self.partitions.itervalues():
            self.result = self.result.union(partition)

    def _create_partitions(self, data, box):
        """
        :type data: pyspark.RDD
        :param data: RDD containing ((key, partition id), k-dim vector
            like)
        :type box: BoundingBox
        :param box: BoundingBox for the entire data set
        """
        current_axis = 0
        todo_q = Queue()
        todo_q.put(0)
        done_q = Queue()
        self.partitions = {0: data}
        self.bounding_boxes = {0: box}
        next_label = 1
        while next_label < self.max_partitions:
            if not todo_q.empty():
                current_label = todo_q.get()
                current_partition = self.partitions[current_label]
                current_box = self.bounding_boxes[current_label]
                if self.split_method == 'min_var':
                    (part1, part2, median), current_axis = min_var_split(
                        current_partition, self.k, next_label)
                else:
                    part1, part2, median = median_search_split(
                        current_partition,
                        current_axis,
                        next_label)
                box1, box2 = current_box.split(current_axis, median)
                self.partitions[current_label] = part1
                self.partitions[next_label] = part2
                self.bounding_boxes[current_label] = box1
                self.bounding_boxes[next_label] = box2
                done_q.put(current_label)
                done_q.put(next_label)
                next_label += 1
            else:
                todo_q = done_q
                done_q = Queue()
                current_axis = (current_axis + 1) % self.k
