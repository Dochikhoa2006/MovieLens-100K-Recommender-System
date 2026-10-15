import numpy as np

class FM_Regression:

    def __init__(self, n_iter = 7, rank = 180, l2_reg = 0.0000000000000000001, step_size = 0.0000000000000000000001, random_state = None):
        self.n_iter = n_iter
        self.rank = rank
        self.l2_reg = l2_reg
        self.step_size = step_size
        self.random_state = random_state
        self.w0 = 0
        self.w = None
        self.V = None

    def fit(self, X, y):
        X = X.tocsr()
            
        n_samples, n_features = X.shape
        self.w0 = np.mean(y) 
        self.w = np.zeros(n_features)

        if self.random_state is None:
            self.V = np.random.normal(scale=0.0001, size=(n_features, self.rank))
        else:
            self.V = np.random.default_rng(self.random_state).normal(
                scale=0.0001, size=(n_features, self.rank)
            )

        for epoch in range(self.n_iter):

            total_squared_error = 0.0
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
                raw_error = y_hat - y[i]
                total_squared_error += raw_error**2
                error = np.clip(raw_error, -5.0, 5.0)

                self.w0 -= self.step_size * error 

                for idx_in_row, global_idx in enumerate(indices):
                    val = values[idx_in_row]
                    
                    grad_w = error * val + self.l2_reg * self.w[global_idx]
                    self.w[global_idx] -= self.step_size * grad_w
                    
                    factors = self.V[global_idx, :]
                    grad_v = val * (sum_v_x - factors * val)
                    update_v = self.step_size * (error * grad_v + self.l2_reg * factors)
                    factors -= np.clip(update_v, -0.1, 0.1)

            current_rmse = np.sqrt(total_squared_error / n_samples)
            print(f'Iteration: {epoch + 1} | RMSE: {current_rmse:.4f}')

            if not np.isfinite(current_rmse) or current_rmse > 100:
                print(f"----- Divergence detected at Epoch {epoch + 1}. Lowering step_size is required -----")
                break

    def predict(self, X):
        X = X.tocsr()

        linear_term = self.w0 + np.asarray(X @ self.w).ravel()
        sum_v_x = np.asarray(X @ self.V)
        sum_sq_v_x = np.asarray(X.multiply(X) @ np.square(self.V))
        interaction_term = 0.5 * np.sum(np.square(sum_v_x) - sum_sq_v_x, axis=1)
        predictions = np.clip(linear_term + interaction_term, 0, 5)

        # Preserve the global-bias prediction for rows with no known features.
        predictions[np.diff(X.indptr) == 0] = self.w0
        return predictions
