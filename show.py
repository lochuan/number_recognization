import numpy
import matplotlib.pyplot
import pylab

data_file = open("train.csv", 'r')
data_list = data_file.readlines()
data_file.close()

num = input("Which image do you want to show?")
all_values = data_list[num].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
pylab.show()
