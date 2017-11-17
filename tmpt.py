
# inspect a mnist image with label
from matplotlib import pyplot as plt

def inspect_mnist(index, images, labels):
  plt.title('Index: {:d}, Label: {:d}'.format(
    index, labels[index].argmax()
  ))
  plt.imshow(images[index][:, :, 0], cmap='gray')
  plt.show()

