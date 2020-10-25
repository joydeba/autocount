import numpy as np
from joblib import load, Parallel, delayed
import sys
import csv

from skimage.feature import peak_local_max
from scipy.ndimage.morphology import distance_transform_edt
from scipy.special import gamma
from skimage.morphology import watershed, remove_small_objects
from skimage import morphology as morph
from skimage.measure import regionprops, label
from skimage.filters import gaussian
from scipy import stats
from scipy import special

import nevergrad as ng

input_file = sys.argv[1]
num_threads = int(sys.argv[3])
labels_file = sys.argv[2]

img_dict = load(input_file) # Segmented input images from dict
num_samples = len(img_dict)



# never used
def csv_to_dict(filename):
    ret = {}

    with open(filename, mode='r') as infile:
        reader = csv.reader(infile)

        for rows in reader:
            ret[rows[0]] = int(rows[1])

    return ret

# never used
def show_performance_stats(predicted, all_y, labels, params):
    for i in range(len(labels)):
        print('{0}: {1} {2} {3}'.format(labels[i], all_y[i], predicted[i], params[i]))

    predicted = np.array(predicted)
    all_y = np.array(all_y)

    all_losses = np.abs(all_y - predicted)
    all_y_mean = np.mean(all_y)
    total_error = np.sum(np.square(all_y - all_y_mean))
    unexplained_error = np.sum(np.square(all_losses))
    r2 = 1. - (unexplained_error / total_error)

    if np.isnan(r2):
        exit()

    print('All losses:')
    print(all_losses)

    print('MAE: {0}'.format(np.mean(all_losses)))

    print('RMSE: {0}'.format(np.sqrt(np.mean(np.square(all_losses)))))

    print('R2: {0}'.format(r2))

    return r2


def do_detection(v, gauss, d_diam, e_diam, thresh):
    v2 = gaussian(v, gauss) #Multi-dimensional Gaussian filter to detect edges.

    image = np.zeros_like(v2) # Return an array of zeros with the same shape and type as a given array
    image[v2 >= thresh] = 1. 
    # label to define unique objects, finds the convex hull of each using convex_hull_image 
    image = morph.dilation(image, morph.disk(d_diam))
    image = morph.erosion(image, morph.disk(e_diam))

    return image

# usses watershed 
def do_segmentation(v, diam, d_diam):
    v = remove_small_objects(v > 0., d_diam)

    distance = distance_transform_edt(v)
    local_maxi = peak_local_max(distance, min_distance=diam, indices=False, labels=v)
    markers = label(local_maxi)
    labels = watershed(-distance, markers, mask=v, watershed_line=False)

    # To prevent min_distance from removing external objects
    labels = label(labels + v)

    return labels


def get_param_count(img, diam, d_diam2):
    '''
        Given an image and parameters the function returns number of objects

        img: binary segmentation mask from the unsupervised segmentation proposed by Kanezaki(2018)
        params:   list containing the parameters to optimize on: watershed marker threshold, diameter for morphological dilation, diameter for morphological erosion
        returns: the number of heads in binary segmenation
    '''

    image = np.pad(img, mode='constant', pad_width=100)

    labels = do_segmentation(image, diam, d_diam2) # with watershed

    return len(regionprops(labels))

# Detect optimal params from loss
def find_detection_params_for_image(v):
    cache = {}

    def detection_loss(gauss, d_diam, e_diam, thresh):
        params = gauss, d_diam, e_diam, thresh

        if params in cache.keys():
            return cache[params]

        image = do_detection(v, gauss, d_diam, e_diam, thresh) # return object edges

        prop = regionprops(label(image)) # regionprops() result to draw certain properties on each region.
        c1 = np.array([p.area for p in prop], dtype=np.float32)

        if len(c1) < 3:
            return np.inf

        # Detection loss

        ratio1 = -np.log(np.sum(v[image > 0.]) / np.sum(v))
        ratio2 = -np.log(np.sum(v[image > 0.]) / float(np.count_nonzero(image)))

        image_loss = ratio1 + ratio2

        cache[params] = image_loss

        return image_loss
    # Instrumentation helps you convert a set of arguments into variables in the data space which can be optimized
    bounds = ng.Instrumentation(ng.var.Scalar(float).bounded(0., 10.),
                                ng.var.Scalar(int).bounded(0, 10),
                                ng.var.Scalar(int).bounded(0, 10),
                                ng.var.Scalar(float).bounded(0., 1.))
    # is excellent in discrete settings of mixed settings when high precision on parameters is not relevant; it's possibly a good choice for hyperparameter choice.
    optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(instrumentation=bounds, budget=4000)
    # Hyper parameter for detection loss and detection
    recommendation = optimizer.minimize(detection_loss)
    opt_params = recommendation.args # optimal params from loss

    return opt_params # params 

# optimized params for all images 
def find_segmentation_params_for_all_images(images):
    def linear_search(loss_fn, range_a, range_b): #linear search
        opt_diam = None
        opt_ddiam = None
        opt_loss = None

        for a in range_a:
            for b in range_b:
                loss = loss_fn(a, b)

                if opt_loss is None or loss < opt_loss:
                    opt_diam = a
                    opt_ddiam = b
                    opt_loss = loss

        return opt_diam, opt_ddiam

    def segmentation_loss(diam, d_diam): 
        def par_loop(v, diam, d_diam):
            def wasserstein(dist1, dist2): # distance 
                bins = np.arange(min(dist1.ppf(0.01), dist2.ppf(0.01)), max(dist1.ppf(0.99), dist2.ppf(0.99)))
                return np.sum(np.abs(dist1.cdf(bins) - dist2.cdf(bins)))

            class ecdf:
                def __init__(self, data):
                    self.lb = min(data)
                    self.ub = max(data)

                    bins = np.arange(self.lb, self.ub + 1)
                    hist, edges = np.histogram(data, bins=bins, density=True)
                    cs = np.cumsum(hist)

                    self.dist = {}

                    for b, c in zip(bins, cs):
                        self.dist[b] = c

                def ppf(self, d):
                    if d == 0.01:
                        return self.lb
                    else:
                        return self.ub

                def cdf(self, loc):
                    ret = []

                    for l in np.round(loc):
                        if l < self.lb:
                            ret.append(0.)
                        elif l >= self.ub:
                            ret.append(1.)
                        else:
                            ret.append(self.dist[int(l)])

                    return np.array(ret)

            v2 = v.copy()
            v2 = np.pad(v2, mode='constant', pad_width=100)

            prop0 = regionprops(label(v2))
            c0 = np.array([p.area for p in prop0], dtype=np.float32)

            labels = do_segmentation(v2, diam, d_diam)

            prop = regionprops(labels)
            c1 = np.array([p.area for p in prop], dtype=np.float32)

            if len(c1) < 3:
                return np.inf

            # Segmentation loss
            a1, loc1, sc1 = stats.gamma.fit(c1)
            m1, s1 = stats.norm.fit(c1)

            g1 = stats.gamma(a1, loc1, sc1)
            norm1 = stats.norm(loc=m1, scale=s1)

            image_loss = wasserstein(norm1, g1) + wasserstein(ecdf(c0), ecdf(c1)) + wasserstein(norm1, ecdf(c1))

            return image_loss

        total_dist = Parallel(n_jobs=num_threads)(delayed(par_loop)(v, diam, d_diam) for v in images)

        return np.mean(np.square(total_dist))

    opt_params = linear_search(segmentation_loss, range(1, 20), range(0, 10))

    return opt_params # optimal params based on linear search + seg loss (wasserstein loss)


# Find all optimal detection parameterizations
print('Doing per-image detection optimization...')
filenames = []
mats = []
counts_dict = {}
params_dict = {}

for f, m in img_dict.items():
    filenames.append(f)
    mats.append(m)
# Mats are paralleled for finding optimal parameters from losses 
all_detect_params = Parallel(n_jobs=num_threads)(delayed(find_detection_params_for_image)(v) for v in mats)

# Find all object counts
print('Doing detections...')
# Using optimized params for object edge detection
all_detected_mats = Parallel(n_jobs=num_threads)(delayed(do_detection)(v, p[0], p[1], p[2], p[3]) for v, p in zip(mats, all_detect_params))

print('Doing global segmentation optimization...')
# Finding optimized params for all together
seg_params = find_segmentation_params_for_all_images(all_detected_mats)
print('Segmentation parameters found: ')
print(seg_params)

print('Counting objects...')

for f, m, opt_params in zip(filenames, all_detected_mats, all_detect_params):
    c = get_param_count(m, seg_params[0], seg_params[1]) # count using watershed segmentation

    counts_dict[f] = c
    params_dict[f] = list(opt_params)

print('Parameters:')
print(params_dict)

print('Object counts:')
print(counts_dict)
