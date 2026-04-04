import numpy as np
from scipy import sparse

class FM_Regression:

    def __init__(self, n_iter = 7, rank = 180, l2_reg = 0.0000000000000000001, step_size = 0.0000000000000000000001):
        self.n_iter = n_iter
        self.rank = rank
        self.l2_reg = l2_reg
        self.step_size = step_size
        self.w0 = 0
        self.w = None
        self.V = None

    def fit(self, X, y):
        if sparse.isspmatrix_csc(X):
            X = X.tocsr()
            
        n_samples, n_features = X.shape
        self.w0 = np.mean(y) 
        self.w = np.zeros(n_features)

        self.V = np.random.normal(scale=0.0001, size=(n_features, self.rank))

        for epoch in range(self.n_iter):

            total_error = 0
            for i in range(n_samples):
                indices = X[i].indices
                values = X[i].data
                if len(indices) == 0: continue

                linear_term = self.w0 + np.dot(self.w[indices], values)
                v_subset = self.V[indices, :]
                sum_v_x = np.dot(values, v_subset) 
                sum_sq_v_x = np.dot(values**2, v_subset**2)
                interaction_term = 0.5 * np.sum(sum_v_x**2 - sum_sq_v_x)
                
                y_hat = linear_term + interaction_term
                error = y_hat - y[i]
                
                error = np.clip(error, -5.0, 5.0)
                total_error += error**2

                self.w0 -= self.step_size * error 

                for idx_in_row, global_idx in enumerate(indices):
                    val = values[idx_in_row]
                    
                    grad_w = error * val + self.l2_reg * self.w[global_idx]
                    self.w[global_idx] -= self.step_size * grad_w
                    
                    for f in range(self.rank):
                        grad_v = val * (sum_v_x[f] - self.V[global_idx, f] * val)
                        update_v = self.step_size * (error * grad_v + self.l2_reg * self.V[global_idx, f])
                        self.V[global_idx, f] -= np.clip(update_v, -0.1, 0.1)

            current_rmse = np.sqrt(total_error / n_samples)
            print(f'Iteration: {epoch + 1} | RMSE: {current_rmse:.4f}')

            if np.isnan(total_error) or current_rmse > 100:
                print(f"----- Divergence detected at Epoch {epoch}. Lowering step_size is required -----")
                break

    def predict(self, X):
        if sparse.isspmatrix_csc(X):
            X = X.tocsr()
            
        predictions = []
        for i in range(X.shape[0]):
            indices = X[i].indices
            values = X[i].data
            
            if len(indices) == 0:
                predictions.append(self.w0)
                continue
            
            linear_term = self.w0 + np.dot(self.w[indices], values)
            v_subset = self.V[indices, :]
            sum_v_x = np.dot(values, v_subset)
            sum_sq_v_x = np.dot(values**2, v_subset**2)
            interaction_term = 0.5 * np.sum(sum_v_x**2 - sum_sq_v_x)
            
            pred = np.clip(linear_term + interaction_term, 0, 5)
            predictions.append(pred)
            
        return np.array(predictions)
    



