from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score

import numpy as np
import pandas as pd


class Evaluate:
    def accuracy(self, y, ypred):

        assert len(y) > 0
        
        unique_pred = np.unique(ypred)
        unique_true = np.unique(y)
        
        Nunique_pred = len(np.unique(ypred))
        Nunique_true = len(np.unique(y))

        # Compute the confusion matrix
        cm = np.zeros((Nunique_pred, Nunique_true), dtype = np.int32)
        for i in range(Nunique_pred):
            for j in range(Nunique_true):
                idx = np.logical_and(ypred == unique_pred[i], y == unique_true[j])
                cm[i][j] = np.count_nonzero(idx)
        
        # Convert the confusion matrix to the cost matrix
        Cmax = np.amax(cm)
        cm = Cmax - cm
        # Get the optimal assignement
        row, col = linear_sum_assignment(cm)

        # Calculate the accuracy from the optimal assignment
        count = 0
        for i in range(Nunique_pred):
            idx = np.logical_and(ypred == unique_pred[row[i]], y == unique_true[col[i]] )
            count += np.count_nonzero(idx)
        
        return 1.0*count/len(y)

    def allMetrics(self, y, ypred, prec = 4):

        acc = self.accuracy(y, ypred)
        nmi = adjusted_mutual_info_score(y, ypred)
        ari = adjusted_rand_score(y, ypred)

        return [round(num, prec) for num in [acc, nmi, ari]]

    def _highlight_max(self, s):
        is_max = s == s.max()
        return ['background-color: palegreen' if v else '' for v in is_max]

    def _highlight_min(self, s):
        is_min = s == s.min()
        return ['background-color: coral' if v else '' for v in is_min]

    # Compare and display results
    def compare_metrics(self, metrics_value, row_names = None, col_names = None, prec = 4):

        values = list()
        for tmp_list in metrics_value:
            values.append(tmp_list[:3]+[np.mean(tmp_list[:3])])
        
        my_col = ["ACC", "NMI", "ARI", "mean"]
        if col_names is not None:
            my_col[:3] = col_names

        my_row = [str(i) for i in range(len(metrics_value))]
        if row_names is not None:
            my_row = row_names
        
        df = pd.DataFrame(values, columns = my_col, index = my_row)
        df = df.sort_values(by ='mean' , ascending=False)

        html = (df.style.format(precision=prec)
                  .apply(self._highlight_min)
                  .apply(self._highlight_max)
                  .set_properties(subset=['mean'], **{'font-weight':'bold'}))
        
        return html



