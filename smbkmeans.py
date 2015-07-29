# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import random


def to_dense(X):
    '''Convert X to dense matrix if necessary.'''
    if isinstance(X, sp.csr_matrix):
        return X.todense()
    else:
        return X


def project_L1(v, l=1., eps=.01):
    '''Perfoms eps-accurate projection of v to L1 ball with radius l.
    Does not preserve the L2 norm.

    See: D. Sculley: "Web-Scale K-Means Clustering",
         Proceedings of the 19th international conference on World Wide Web,
         (2010)
    '''
    L1 = np.linalg.norm(v, ord=1)

    if L1 <= l * (1. + eps):
        return v

    upper = np.linalg.norm(v, ord=np.inf)
    lower = 0.
    theta = 0.
    current = L1

    while (current > l * (1. + eps)) or (current < l):
        theta = .5 * (upper + lower)
        current = np.maximum(
                0.,
                np.add(np.fabs(v), -theta)).sum()

        if current <= l:
            upper = theta
        else:
            lower = theta

    return np.multiply(np.sign(v), np.maximum(0., np.add(np.fabs(v), -theta)))


def to_sparse(X, project=False, inplace=True, l=1., eps=.01):
    '''Convert X to sparse matrix if necessary.

    On request perfoms eps-accurate projection to L1 ball with radius l.
    If inplace=True, this will overwrite X in-place.
    '''
    if isinstance(X, sp.csr_matrix):
        if inplace is False:
            X = X.copy()

        if project is True:
            n_rows, _ = X.get_shape()
            for row_idx in range(n_rows):
                # project on L1 ball
                X.data[X.indptr[row_idx]:X.indptr[row_idx + 1]] = project_L1(
                        v=X.data[X.indptr[row_idx]:X.indptr[row_idx + 1]],
                        l=l,
                        eps=eps)
                # length-normalize
                X.data[X.indptr[row_idx]:X.indptr[row_idx + 1]] /= np.linalg.norm(
                        X.data[X.indptr[row_idx]:X.indptr[row_idx + 1]])

            # make sparse representation more efficient
            # (inplace operation)
            X.eliminate_zeros()

        return X
    else:
        if inplace is False:
            X = np.array(X)

        if project is True:
            n_rows, _ = X.shape
            for row_idx in range(n_rows):
                # project on L1 ball
                X[row_idx] = project_L1(
                        v=X[row_idx],
                        l=l,
                        eps=eps)
                # length-normalize
                X[row_idx] /= np.linalg.norm(
                        X[row_idx])

        return sp.csr_matrix(X)


def cosine_distances(X, Y):
    '''Return a cosine distance matrix.

    Computes the cosine distance matrix between each pair of vectors in the
    rows of X and Y. These vectors must have unit length.
    '''
    # compute cosine similarities
    # (we use *, because np.dot is not aware of sparse)
    if isinstance(X, sp.csr_matrix) or isinstance(Y, sp.csr_matrix):
        dist = X * Y.T

        if isinstance(dist, sp.csr_matrix):
            dist = to_dense(dist)
    else:
        dist = np.dot(X, Y.T)

    # convert into cosine distances
    dist *= -1.
    dist += 1.

    # make sure that all entries are non-negative
    np.maximum(dist, 0., out=dist)

    return dist


class SphericalMiniBatchKMeans:
    def __init__(self,
                 n_clusters=3,
                 n_init=3,
                 max_iter=100,
                 batch_size=100,
                 max_no_improvement=10,
                 reassignment_ratio=.01,
                 project_l=1.,
                 project_eps=.01):
        '''Set instance parameters.'''
        self.n_clusters = n_clusters

        self.max_iter = max_iter
        self.n_init = n_init

        self.batch_size = batch_size

        self.max_no_improvement = max_no_improvement
        self.reassignment_ratio = reassignment_ratio

        self.project_l = project_l
        self.project_eps = project_eps

        self.inertia_ = None
        self.centers_ = None
        self.counts_ = None

    def fit(self, n_samples, get_batch):
        '''Compute the cluster centers by chunking the data into mini-batches.

        Adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/k_means_.py.

        TODO: figure out if we need to cache any distances
              or if having a single distances array is beneficial from a
              performance/memory utilization point of view.
        '''

        n_batches = np.int64(
                np.ceil(np.float64(n_samples) / self.batch_size))

        n_init = self.n_init
        init_batch_size = min(
                3 * self.batch_size, n_samples)

        # sample validation batch for initial clustering
        init_validation_batch = get_batch(
                batch_size=init_batch_size)

        # look repeatedly for the best possible initial clustering
        for init_idx in xrange(n_init):
            # set initial counts to zero
            init_counts = np.zeros(
                    self.n_clusters,
                    dtype=np.int32)

            # sample initial batch
            init_batch = get_batch(batch_size=init_batch_size)

            # come up with initial centers (CSR format)
            init_centers = self._init(batch=init_batch)

            nearest_center, _ = self._labels_inertia(
                    batch=init_validation_batch,
                    centers=init_centers,
                    compute_labels=True)

            # refine centers and get their counts
            dense_init_centers = to_dense(init_centers)
            self._update(
                    batch=init_validation_batch,
                    nearest_center=nearest_center,
                    counts=init_counts,
                    dense_centers=dense_init_centers)
            init_centers = to_sparse(
                    dense_init_centers,
                    project=True,
                    inplace=True,
                    l=self.project_l,
                    eps=self.project_eps)

            # get the inertia of the refined centers
            _, inertia = self._labels_inertia(
                    batch=init_validation_batch,
                    centers=init_centers,
                    compute_labels=False)

            # identify the best initial guess
            if (self.inertia_ is None) or (inertia < self.inertia_):
                self.centers_ = init_centers
                self.counts_ = init_counts
                self.inertia_ = inertia

        # context to be used by the _convergence() routine
        convergence_context = {}

        # convert cluster centers to dense format
        dense_centers = to_dense(self.centers_)

        # optimize the clustering iteratively until convergence
        # or maximum number of iterations is reached
        n_iter = np.int64(self.max_iter * n_batches)
        for iter_idx in xrange(n_iter):
            # sample a mini-batch by picking randomly from data set
            mini_batch = get_batch(self.batch_size)

            # randomly reassign?
            if (((iter_idx + 1)
                 % (10 + self.counts_.min()) == 0)) and (self.reassignment_ratio > 0.):
                self._reassign(
                        batch=mini_batch,
                        counts=self.counts_,
                        dense_centers=dense_centers,
                        reassignment_ratio=self.reassignment_ratio)

            # convert cluster centers to sparse format
            # and perfom an eps-accurate projection to an L1 ball
            # to reduce the number of non-zero components
            self.centers_ = to_sparse(
                    dense_centers,
                    project=True,
                    inplace=True,
                    l=self.project_l,
                    eps=self.project_eps)

            # find nearest cluster centers for data in mini-batch,
            # i.e. find concept vectors closest in cosine similarities
            nearest_center, self.inertia_ = self._labels_inertia(
                    batch=mini_batch,
                    centers=self.centers_,
                    compute_labels=True)

            # test convergence
            if (self._convergence(n_samples,
                                  context=convergence_context,
                                  batch_inertia=self.inertia_)):
                break

            # incremental update of the cluster centers, i.e. the concept
            # vectors.
            dense_centers = to_dense(init_centers)
            self._update(batch=mini_batch,
                         nearest_center=nearest_center,
                         counts=self.counts_,
                         dense_centers=dense_centers)

        self.n_iter_ = iter_idx + 1

        return self

    def _init(self, batch):
        '''Selects initial centers in a smart way to speed up convergence.
        see: Arthur, D. and Vassilvitskii, S.,
             "k-means++: the advantages of careful seeding",
             ACM-SIAM symposium on Discrete algorithms (2007)

        Adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/k_means_.py.
        '''
        n_samples = batch.shape[0]
        n_centers = self.n_clusters
        n_features = batch.shape[1]

        # set the number of local seeding trials
        n_local_trials = 2 + np.int64(np.log(n_centers))

        # pick first center randomly
        centers = np.empty((n_centers,), dtype=np.int32)
        centers[0] = random.randint(0, n_samples-1)
        # initialize the list of closest distances
        closest_dist = cosine_distances(batch[centers[0]], batch).getA1()
        # calculate the sum of the distances
        current_dist_sum = closest_dist.sum()

        # estimate the remaining n_centers-1 centers
        for c in xrange(1, n_centers):
            # sample center candidates with a probability proportional to the
            # cosine distance to the closest existing center (such that centers
            # least similar to the existing centers are more likely to be
            # drawn)
            candidate_ids = np.searchsorted(
                    closest_dist.cumsum(),
                    [random.random() * current_dist_sum for _ in range(n_local_trials)])

            # compute distances to center candidates
            candidate_dist = cosine_distances(batch[candidate_ids], batch)

            # decide which candidate is the best
            best_candidate = None
            best_dist = None
            best_dist_sum = None
            for trial in xrange(n_local_trials):
                # element-wise comparison
                new_dist = np.minimum(closest_dist,
                                      candidate_dist[trial].getA1())
                new_dist_sum = new_dist.sum()

                # identify the best local trial
                if (best_candidate is None) or (new_dist_sum < best_dist_sum):
                    best_candidate = candidate_ids[trial]
                    best_dist = new_dist
                    best_dist_sum = new_dist_sum

            # assign center to best local trial
            centers[c] = best_candidate
            # update the list of closest distances
            closest_dist = best_dist
            # update the current distance sum
            current_dist_sum = best_dist_sum

        return to_sparse(X=batch[centers],
                         project=True,
                         inplace=True,
                         l=self.project_l,
                         eps=self.project_eps)

    def _reassign(self, batch, counts, dense_centers, reassignment_ratio=None):
        '''Reassign centers that have very low counts.

        Adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/k_means_.py.
        '''
        n_samples, _ = batch.get_shape()

        if reassignment_ratio is None:
            reassignment_ratio = self.reassignment_ratio

        to_reassign = np.float64(counts) < reassignment_ratio * np.float64(counts.max())
        n_reassigns = to_reassign.sum()

        # pick at most 0.5 * n_samples new centers
        max_reassigns = np.int64(.5 * np.float64(n_samples))
        if n_reassigns > max_reassigns:
            to_reassign[np.argsort(counts)[max_reassigns:]] = False
            n_reassigns = to_reassign.sum()

        if n_reassigns:
            # pick new centers amongst observations with uniform probability
            new_centers = np.random.choice(n_samples,
                                           n_reassigns)

            dense_centers[to_reassign] = batch[new_centers].todense()

            # a dirty hack from scikit-learn that resets the counts somewhat
            counts[to_reassign] = np.min(counts[~to_reassign])

    def _update(self, batch, nearest_center, counts, dense_centers):
        '''Update cluster centers.

        Adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/k_means_.py.
        '''
        n_centers = self.n_clusters

        batch_data = batch.data
        batch_indices = batch.indices
        batch_indptr = batch.indptr

        for center_idx in xrange(n_centers):
            # find records in mini-batch assigned to this center
            center_mask = nearest_center == center_idx

            count = center_mask.sum()

            if count > 0:
                # the new center will be a convex sum of
                # (old count / new count) * center
                # and (1 - old count / new count) * mean,
                # where mean is the average of all samples that
                # are closest to the center

                # multiply by the old count
                dense_centers[center_idx] *= counts[center_idx]

                # add (new count - old count) * mean
                for sample_idx, nearest in enumerate(center_mask):
                    if nearest:
                        # element-wise addition of the samples
                        for col_idx in xrange(batch_indptr[sample_idx],
                                              batch_indptr[sample_idx + 1]):
                            dense_centers[center_idx, batch_indices[col_idx]] += batch_data[col_idx]

                # update count
                counts[center_idx] += count
                # divide by the new count
                dense_centers[center_idx] /= counts[center_idx]
                # project center onto the unit sphere such that it mimicks a
                # concept vector
                dense_centers[center_idx] /= np.linalg.norm(
                        dense_centers[center_idx], ord=2)

    def _convergence(self, n_samples, context, batch_inertia=None):
        '''Early stopping logic.

        Adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/k_means_.py.
        '''
        if batch_inertia is None:
            batch_inertia = self.inertia_
        batch_inertia /= self.batch_size

        # use an exponentially weighted average (ewa) of the inertia to monitor
        # the convergence
        ewa_inertia = context.get('ewa_inertia')
        if ewa_inertia is None:
            ewa_inertia = batch_inertia
        else:
            alpha = np.float64(self.batch_size) * 2. / (
                    np.float64(n_samples) + 1.)
            alpha = 1. if alpha > 1. else alpha
            ewa_inertia = ewa_inertia * (1. - alpha) + batch_inertia * alpha

        # early stopping heuristic due to lack of improvement on smoothed
        # inertia
        ewa_inertia_min = context.get('ewa_inertia_min')
        no_improvement = context.get('no_improvement', 0)
        if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
            no_improvement = 0
            ewa_inertia_min = ewa_inertia
        else:
            no_improvement += 1

        if self.max_no_improvement <= no_improvement:
            return True

        # update the convergence context to maintain state across successive
        # calls:
        context['ewa_inertia'] = ewa_inertia
        context['ewa_inertia_min'] = ewa_inertia_min
        context['no_improvement'] = no_improvement

        return False

    def _labels_inertia(self, batch, centers=None, compute_labels=False):
        if centers is None:
            centers = self.centers_

        dist = cosine_distances(centers, batch)

        if compute_labels is True:
            labels = dist.argmin(axis=0).getA1()
            inertia = dist[labels, np.arange(dist.shape[1])].sum()
        else:
            labels = None
            inertia = dist.min(axis=0).sum()

        return labels, inertia

    def predict(self, batch):
        '''Predict labels.

        TODO: check if model is fitted.
        '''

        batch_labels, batch_inertia = self._labels_inertia(
                batch=batch,
                compute_labels=True)

        return batch_labels, batch_inertia
