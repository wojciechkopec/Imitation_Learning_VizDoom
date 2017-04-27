import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import numpy as np

file_path = "dropout_groups_means.csv"
plot_name = "Dokladnosc klasyfikacji - Dropout"
x_par = "dropout test"
y_par = "dropout trening"
z_par = "dokladnosc"

# file_path = "bootstrap_groups_variances.csv"
# plot_name = "Odchylenie jakosci rozdzialu - Bootstrap"
# x_par = "glowy"
# y_par = "prawd."
# z_par = "wspolczynnik"


a_list = []

with open(file_path) as file:
    data_iter = csv.reader(file,
                           delimiter=',',
                           quotechar='"')
    data = [d for d in data_iter]
    labels = data[0]
    x_col = labels.index(x_par)
    y_col = labels.index(y_par)
    z_col = labels.index(z_par)
    for line in data[1:]:
        a_list.append([(float(line[x_col]), float(line[y_col])), float(line[z_col])])
dict_value = {x: y for x, y in a_list}

x_labels = list(set(([x[0] for x in dict_value])))
x_labels.sort()
plt_x = np.asarray(x_labels)

y_labels = list(set(([y[1] for y in dict_value])))
y_labels.sort()
plt_y = np.asarray(y_labels)
plt_z = np.zeros(shape=(len(plt_x), len(plt_y)))

for i_x in range(len((plt_x))):
    for i_y in range(len((plt_y))):
        plt_z[i_x][i_y] = dict_value[(plt_x.item(i_x), plt_y.item(i_y))]


plt.imshow(plt_z, cmap=plt.cm.Blues, interpolation='nearest')
plt.axis([0 -0.5, len(plt_y) -0.5, 0 -0.5, len(plt_x) -0.5])
plt.title(plot_name)
plt.xlabel(y_par)
plt.ylabel(x_par)
plt.colorbar().set_label(z_par)
ax = plt.gca()
plt.xticks(range(len(plt_y)), y_labels)
plt.yticks(range(len(plt_x)), x_labels)
plt.show()