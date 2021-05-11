import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Initialize 5 (2560x3) random matrices where each row represents one particular lineage
random_vector_list = [np.random.rand(2560, 3) for i in range(5)]

# Find the row sum of each matrix
sum_random_vector_list = [matrix.sum(axis=1) for matrix in random_vector_list]

# Normalize each row in all matrices by dividing each row with its row sum
normalized_random_vector_list = [matrix/sum[:, np.newaxis] for matrix, sum in zip(random_vector_list, sum_random_vector_list)]

# Coordinate of right triangle
top_right_coord = (10, 10)
top_left_coord = (0, 10)
bottom_left_coord = (0, 0)

triangle_vertices = np.array([top_right_coord, top_left_coord, bottom_left_coord])

# Converting ternary coordinate -> cartesian coordinate in all matrices using matrix multiplication
cartesian_random_vector_list = [np.dot(matrix, triangle_vertices) for matrix in normalized_random_vector_list]

# Find the Euclidean distance between two adjacent timepoints
difference_list = [(cartesian_random_vector_list[i] - cartesian_random_vector_list[i+1])**2 for i in range(len(cartesian_random_vector_list)-1)]
motility_list = [np.sqrt(np.dot(matrix, [1, 1])) for matrix in difference_list]

# Find the motility
for timepoint in range(3):
    fig, ax = plt.subplots()

    x = motility_list[timepoint]
    y = motility_list[timepoint+1]


    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    rho, p_ = scipy.stats.spearmanr(x, y)

    ax.plot(np.arange(0, max(x)), intercept + slope * np.arange(0, max(x)), color='red')
    ax.scatter(x, y, marker='.', color='blue')
    print('transition, timepoint {}, r {}'.format(timepoint, round(r, 3)))
    ax.set_ylim(0, max(y) + 0.05)
    ax.set_xlim(0, max(x) + 0.05)

    plt.show()