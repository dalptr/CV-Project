import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm


def main():
    x, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,
                                        n_clusters_per_class=1,
                                        random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='coolwarm')
    current_axes = plt.gca()
    xlim = current_axes.get_xlim()
    ylim = current_axes.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Support Vector Machine')
    plt.show()

    print("Accuracy on training set: {:.3f}".format(classifier.score(x_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(classifier.score(x_test, y_test)))


main()
