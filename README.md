# IBkLG
Instance Based K-nearest using Log and Gaussian weight kernels

In K-NN distance can be weighted distance such as Inverse of the distance or based on similarity. Similarly, here one can associate weights based on the negative logarithm or another intuitive way is to associate a gaussian. A gaussian is assumed around each and every K-Nearest neighbor and weights are associated relative to the distance of the neighbor from the mean in the gaussian. This package extents the base IBk class to add these two kernels to K-NN algorithm.
