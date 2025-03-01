from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
digits = load_digits()
print(digits.data.shape)
print(digits.target)   # actual labels of the images
plt.gray()
for k in range(5):
    plt.matshow(digits.images[k])
plt.show()
print(digits.target[0])


import numpy as np
image_data = digits.images
n_samples = len(image_data)
flatten_images = image_data.reshape((n_samples, -1))
flatten_images.shape

def kmeans_plus(X, k):
    # Select the first cluster center randomly
    centers = [X[np.random.choice(X.shape[0], 1, replace=False)][0]]

    # Select the remaining cluster centers using k-means++
    for i in range(1, k):
        # Compute the distances to the nearest cluster center for each data point
        tempX = X[:, np.newaxis, :]
        distances = np.min(np.sum((tempX - centers)**2, axis=2), axis=1)

        # Compute the probabilities of each data point being selected as the next cluster center
        probs = distances / np.sum(distances)

        # Select the next cluster center randomly, weighted by the probabilities
        centers.append(X[np.random.choice(X.shape[0], 1, replace=False, p=probs)][0])

    return np.array(centers)
Mu = kmeans_plus(flatten_images, 3)   
Mu   # print the initial centers with k = 3


def manhattan(x, y):
    return (abs(x[0]-y[0]) + abs(x[1]-y[1]))
import numpy as np
X = np.array([[1, 1], [0, 1], [1, 0], [2, 1], [1, 2], [3, 2], [2, 3], [4, 10], [10, 4], [5, 5]])
n = 10   # num points
list_iter = [10, 15, 20, 25, 50, 100]   # num iterations list
## YOUR SOLUTION

for t in list_iter:
    random_errors = []
    kmeans_plus_errors = []
    
    for u in range(100):
        random_centers = X[np.random.choice(X.shape[0],2,replace=False)]
        random_error = sum(min(manhattan(x,c) for c in random_centers) for x in X)
        random_errors.append(random_error)
        
        kmeans_plus_centers = kmeans_plus(X,2)
        kmeans_plus_error = sum(min(manhattan(x,c) for c in kmeans_plus_centers) for x in X)
        kmeans_plus_errors.append(kmeans_plus_error)
        
    better_count = sum(kp < r for kp,r in zip(kmeans_plus_errors, random_errors))
    probability = better_count/100
    print(f"probability of better kmeans for t={t} is {probability}")
    
