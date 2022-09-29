import numpy as np


class GradientDescent:

    def __init__(self, cost_function, params, hyperparameters):

        self.cost_function = cost_function
        self.params = np.array(params)

        self.hyperparameters = hyperparameters
        self.learning_rate = self.hyperparameters.get('lr', 1e-3)
        self.h = self.hyperparameters.get('h', 0.01)
        self.max_iter = self.hyperparameters('max_iter', 100)
        self.min_delta = self.hyperparameters('min_delta', 1e-4)

    def get_jacobian(self):

        cost = self.cost_function(self.params, self.params)

        jacobian = []
        for i in range(len(self.params)):

            mod_params = self.params.copy()
            mod_params[i] = mod_params[i] + self.h

            cost_h = self.cost_function(mod_params, self.params)

            partial_derivative = (cost_h[0] - cost[0]) / self.h

            jacobian.append(partial_derivative)

        jacobian = np.array(jacobian)
        jacobian = np.clip(jacobian, -10, 10)  # Gradient clipping

        return jacobian

    def update_parameters(self):

        jacobian = self.get_jacobian()
        self.params = self.params - self.learning_rate * jacobian

        return self.params

    def optimize(self):

        history = []
        prev_cost = 1e10

        for i in range(self.max_iter):

            if i > 0 and i % 1 == 0:
                print('Iteration {}'.format(i))
                print('\tCurrent Cost = {}'.format(prev_cost))

            self.update_parameters()
            new_cost, ref, pred, dom = self.cost_function(self.params, self.params)

            cost_change = abs(prev_cost - new_cost)
            if cost_change < self.min_delta:
                print('Cost change is below target.')
                print('Cost change = {} - {} = {}'.format(
                    prev_cost, new_cost, cost_change
                ))
                print('Triggering early stop on iteration {}.'.format(i))
                break

            prev_cost = new_cost
            history.append((self.params, new_cost))

        return self.params, history
