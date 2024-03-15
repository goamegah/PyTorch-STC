import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score


 # Calculer le NMI
nmi = normalized_mutual_info_score

def acc(y_true, y_pred):
    # Vérifier que les deux tableaux ont la même taille
    assert len(y_pred) == len(y_true), "Les tableaux y_pred et y_true doivent avoir la même taille."

    # Nombre de classes
    num_classes = max(max(y_pred), max(y_true)) + 1

    # Initialiser la matrice de confusion
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Remplir la matrice de confusion
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[pred_label, true_label] += 1

    # Trouver la correspondance optimale entre les étiquettes prédites et les étiquettes réelles
    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)

    # Calculer la précision en fonction de la correspondance optimale
    total_correct = sum([confusion_matrix[row, col] for row, col in zip(row_ind, col_ind)])
    accuracy = total_correct / len(y_pred)

    return accuracy


def purity_score(y_true, y_pred):
    # compute confusion matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cal_metric(y_pred, y_true):
    s1 = mutual_info_score(y_pred, y_true)
    print(f'mutual_info_score = {s1}')
    s2 = normalized_mutual_info_score(y_pred, y_true)
    print(f'normalized_mutual_info_score = {s2}')
    s3 = adjusted_rand_score(y_pred, y_true)
    print(f'adjusted_rand_score = {s3}')
    s4 = purity_score(y_true, y_pred)
    print(f'purity = {s4}')
    return [s1, s2, s3, s4]