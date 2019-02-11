import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv

scaler = StandardScaler()

a = np.array([
    [10.5, 8, 12, 12, 12],
    [17, 14, 13, 9, 17],
    [5, 6.5, 10.5, 10.5, 9],
    [19, 14.5, 17, 15, 18],
    [12.5, 14, 15, 16, 9],
    [7.5, 9, 17, 15, 15],
    [13, 13, 15, 14.5, 12],
    [8, 13, 12.5, 15, 10.5],
    [0, 2, 10.5, 6, 8.5],
    [19, 18, 15, 15, 18],
    [10.5, 5, 16.5, 18, 7.5]
])

scaler.fit(a)
#a = scaler.transform(a)
a -= np.mean(a, axis=0)
print(a.mean(axis = 0))
print(a.std(axis = 0))
print(np.corrcoef(a.transpose()))

pca = PCA(n_components=5)
result = pca.fit_transform(a)


print("components")
print(pca.components_)
print(pca.singular_values_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid()
plt.show()
print(result)
print(np.matmul(a, np.transpose(pca.components_)))

i = 0
for elt in result:
    plt.scatter(elt[0], elt[1])
    plt.annotate("Elève " + str(i), xy=(elt[0], elt[1]))
    i+=1
plt.show()

axis_names = ["Maths", "Physique", "Anglais", "Français", "SVT"]
for i in range(pca.components_.shape[-1]):
    plt.arrow(0, 0, pca.components_[0][i]*10, pca.components_[1][i]*10)
    plt.annotate(axis_names[i], xy=(pca.components_[0][i]*10, pca.components_[1][i]*10))

i = 0
for elt in result:
    plt.scatter(elt[0], elt[1])
    plt.annotate("Elève " + str(i), xy=(elt[0], elt[1]))
    i+=1

plt.axis([-20, 20, -10, 10])
plt.show()


