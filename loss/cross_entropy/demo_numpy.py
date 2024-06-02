import numpy as np


def cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred))


y = np.array([1, 0, 0])
y_good_pred = np.array([0.8, 0.1, 0.05])
y_bad_pred = np.array([0.2, 0.7, 0.9])

print(f"Good prediction cross entropy is {cross_entropy(y, y_good_pred)}")
print(f"Bad prediction cross entropy is {cross_entropy(y, y_bad_pred)}")
