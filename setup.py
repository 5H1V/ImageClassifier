from skimage.io import imread
import matplotlib.pyplot as plt
image = imread('./image.png')
plt.imshow(image)
plt.show()
