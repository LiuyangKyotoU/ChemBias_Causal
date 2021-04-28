import math


class Evaluator:

    def _mae(self, x, y, std):
        return (x * std - y * std).abs().sum().item()

    def _rmse(self, x, y, std):
        return ((x * std - y * std) ** 2).sum().item()

    def _keep(self, x):
        return x

    def _sqrt(self, x):
        return math.sqrt(x)

    def _mae_funcs(self):
        return self._mae, self._keep

    def _rmse_funcs(self):
        return self._rmse, self._sqrt

    def get_error_func(self, task):
        if task[:3] == 'qm9':
            return self._mae_funcs()
        if task == 'zinc':
            return self._mae_funcs()
        if task == 'esol':
            return self._rmse_funcs()
        if task == 'lipo':
            return self._rmse_funcs()
        if task == 'freesolv':
            return self._rmse_funcs()
