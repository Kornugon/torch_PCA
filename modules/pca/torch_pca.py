from typing import Optional

import torch
import plotly.graph_objs as go


class PCA:
    """
    Class of Principal Component Analysis in PyTorch
    using Singular Value Decomposition (SVD).
    """
    _types_validation = {
        "device": str,
        "n_components": int
    }

    def __init__(
            self,
            device: str = 'cpu',
            n_components: Optional[int] = None) -> None:
        """
        :param str device: Selected compute device from [`cpu`, `cuda`].\
            Default: 'cpu'.
        :param Optional[int] n_components: The number of principal components.\
            If not given, n_components will be chosen automatically.\
            Default: None.
        """
        self._validate_args(
            device=device,
            n_components=n_components
        )

        self.device = torch.device(device)
        self.n_components = n_components

        self.V = None
        self.optimal_n = None
        self._ratio_orig = None
        self._explained_variance = None
        self._explained_variance_ratio = None

    def fit(
            self,
            data: torch.Tensor,
            full_matrices: bool = False,
            expected_ratio: float = 0.98,
            offset_ratio: float = 0.02) -> None:
        """
        Fits the PCA model to the given data.

        :param torch.Tensor data: The input data as a torch.Tensor of shape\
            (n, m), where n is the number of samples and m is\
            the number of features.
        :param bool full_matrices: Whether or not to compute the full U and V\
            matrices in the SVD decomposition. Default: False.
        :param float expected_ratio: Desired 'explained_variance_ratio'.\
            Has effect when n_components is not given with __init__.\
            Default: 0.98.
        :param float offset_ratio: Offset for 'explained_variance_ratio'.\
            Has effect when n_components is not given with __init__.\
            Default: 0.02.
        """
        self._validate_altern_args(
            full_matrices,
            expected_ratio,
            offset_ratio
        )

        if self.n_components:
            self._fit(data, full_matrices)
            self.fit_params = {
                "device": self.device,
                "n_components": self.n_components,
                "full_matrices": full_matrices
            }
        else:
            self.n_components = data.shape[1]
            self._fit(data, full_matrices)
            self._ratio_orig = self._explained_variance_ratio
            self.n_components = self._optimal_n_components(
                self._ratio_orig,
                expected_ratio,
                offset_ratio
            )
            self._fit(data, full_matrices)
            self.fit_params = {
                "device": self.device,
                "n_components": self.n_components,
                "full_matrices": full_matrices,
                'expected_ratio': expected_ratio,
                'offset_ratio': offset_ratio
            }

        self._explained_variance = self._explained_variance[
            :self.n_components]
        self._explained_variance_ratio = self._explained_variance_ratio[
            :self.n_components]

        explanation = round((torch.cumsum(
            self._explained_variance_ratio, dim=0)[-1].item()) * 100, 2)

        exp_info = f"About {explanation}% of data would be explained"
        exp_info += f" by setting PCA(n_components={self.n_components})"
        print(exp_info)

    def transform(
            self,
            data: torch.Tensor) -> torch.Tensor:
        """
        Transforms the given data using the fitted PCA model.

        :param torch.Tensor data: The input data as a torch.Tensor of shape\
            (n, m), where n is the number of samples and m is\
            the number of features.

        :return torch.Tensor: The projection of the data onto the first\
            n_components principal components as a torch.Tensor\
            of shape (n, n_components).
        """
        if self.V is None:
            raise ValueError("The PCA model has not been fitted")

        data = data.to(self.device).float()
        data = data - data.mean(dim=0)

        return torch.matmul(data, self.V).cpu()

    def fit_transform(
            self,
            data: torch.Tensor,
            full_matrices: bool = False,
            expected_ratio: float = 0.98,
            offset_ratio: float = 0.02) -> torch.Tensor:
        """
        Fits the PCA model to the given data and then transforms the data
        using the fitted model.

        :param torch.Tensor data: The input data as a torch.Tensor of shape\
            (n, m), where n is the number of samples and m is\
            the number of features.
        :param bool full_matrices: Whether or not to compute the full U and V\
            matrices in the SVD decomposition.
        :param float expected_ratio: Desired 'explained_variance_ratio'.\
            Has effect when n_components is not given with __init__.\
            Default: 0.98.
        :param float offset_ratio: Offset for 'explained_variance_ratio'.\
            Has effect when n_components is not given with __init__.\
            Default: 0.02.

        :return torch.Tensor: The projection of the data onto the first\
            n_components principal components as a torch.Tensor\
            of shape (n, n_components).
        """
        self.fit(
            data,
            full_matrices=full_matrices,
            expected_ratio=expected_ratio,
            offset_ratio=offset_ratio
        )

        return self.transform(data)

    def plot_explained_variance(
            self,
            width: int = 850,
            height: int = 550) -> None:
        """
        Plotting explained variance ratio.

        :param int width: The width of the plot in pixels. Default: 850.
        :param int height: The height of the plot in pixels. Default: 550.
        """
        title = 'Explained variance cumulative sum'
        trace_0_title = 'variance'

        if self._ratio_orig is not None:
            optimal_point = self.optimal_n
            ratio = self._ratio_orig.cpu()
            trace_1_title = 'optimal n_components'
        else:
            ratio = self._explained_variance_ratio.cpu()
            optimal_point = self.n_components
            trace_1_title = 'given n_components'

        layout = go.Layout(
            xaxis_title="number of components",
            yaxis_title="cumulative explained variance",
            width=width,
            height=height,
            xaxis=dict(tickangle=-45, tickvals=torch.arange(
                0, len(ratio), 1 if len(ratio) < 20 else 2)),
            title=dict(text=title, x=0.5))

        fig = go.Figure(
            layout=layout,
            data=[go.Scatter(x=torch.arange(1, len(ratio) + 1),
                             y=torch.cumsum(ratio, dim=0),
                             mode='markers+lines',
                             name=trace_0_title),
                  go.Scatter(x=[optimal_point],
                             y=[torch.cumsum(ratio, dim=0)[optimal_point - 1]],
                             mode='markers',
                             marker=dict(size=8, color='red'),
                             name=trace_1_title)])
        fig.show()

    def _fit(
            self,
            data: torch.Tensor,
            full_matrices: bool = False) -> None:
        """
        Fits the PCA model to the given data.

        :param bool full_matrices: Controls whether to compute\
            the full or reduced SVD. Default: False.
        """
        data = data.to(self.device).float()
        data = data - data.mean(dim=0)

        # Compute the SVD (U, S, V) of the centered data
        _, S, V = torch.linalg.svd(data, full_matrices=full_matrices)

        self.V = V[:, :self.n_components]
        self._explained_variance = torch.div(torch.pow(S, 2),
                                             (data.shape[0] - 1))
        total_variance = torch.sum(self._explained_variance)
        self._explained_variance_ratio = torch.div(self._explained_variance,
                                                   total_variance)

    def _optimal_n_components(
            self,
            explained_var_ratio: torch.Tensor,
            expected_ratio: float = 0.98,
            offset_ratio: float = 0.02) -> int:
        """
        Get optimal 'n_components' for PCA, based on
        'expected_ratio' of 'explained_variance_ratio'.
        Function is called only when n_components is not given with __init__.

        :param torch.Tensor explained_var_ratio: Input data tensor\
            with explained_variance_ratio for each component.
        :param float expected_ratio: Desired 'explained_variance_ratio'.\
            Has effect when n_components is not given with __init__.
        :param float offset_ratio: Offset for 'explained_variance_ratio'.\
            Has effect when n_components is not given with __init__.

        :raise RangeCriteriaException: Raise exception if user set\
            'expected_ratio' or 'offset_ratio' parameter, too low.

        :return int: First element of list of 'n_components',\
            that fits criteria of 'expected_ratio', eg. PCA(n_components).
        """
        cumsum_ratio = torch.cumsum(explained_var_ratio, dim=0)
        lower_ratio_limit = expected_ratio - offset_ratio
        upper_ratio_limit = expected_ratio + offset_ratio
        components_num = []

        for i, ratio in enumerate(cumsum_ratio, start=1):
            if lower_ratio_limit <= ratio <= upper_ratio_limit:
                components_num.append(i)

        if components_num:
            self.optimal_n = components_num[0]
        else:
            raise RangeCriteriaException(
                "Expected ratio criteria, out of range.")

        return self.optimal_n

    def _validate_args(self, **args) -> None:
        """
        Arguments validation for PCA model.

        :params dict[str, Any] **args: Model parameters dict.
        """
        device_options = ('cpu', 'cuda')
        for arg, arg_value in args.items():
            true_arg_type = self._types_validation.get(arg, None)

            if true_arg_type is None:
                raise KeyError(
                    f"""Argument {arg} doesn't exist in the
                    class variable `_types_validation`.""")

            if arg_value is not None and not isinstance(arg_value,
                                                        true_arg_type):
                raise TypeError(
                    f"""Argument {arg} must be of type {true_arg_type}
                    (actual: {type(arg_value)}).""")

            if args['device'] not in device_options:
                raise ValueError(
                    f"""Parameter 'device' has to be one of: {device_options}.
                    Received device = {args['device']}.""")

            if args['n_components'] is not None and args['n_components'] <= 0:
                raise ValueError(
                    f"""Parameter 'n_components' has to be positive integer.
                    Received n_components = {args['n_components']}.""")

    def _validate_altern_args(
            self,
            full_matrices: bool,
            expected_ratio: float,
            offset_ratio: float) -> None:
        """
        Arguments validation for fit method.
        """
        if not isinstance(full_matrices, bool):
            raise TypeError(
                f"""Argument 'full_matrices' must be of type bool
                (actual: {type(full_matrices)}).""")

        if self.n_components is None:
            if not isinstance(expected_ratio, float):
                raise TypeError(
                    f"""Argument 'expected_ratio' must be of type float
                    (actual: {type(expected_ratio)}).""")

            if not isinstance(offset_ratio, float):
                raise TypeError(
                    f"""Argument 'offset_ratio' must be of type float
                    (actual: {type(offset_ratio)}).""")

            if expected_ratio <= 0 or expected_ratio > 1:
                raise ValueError(
                    f"""Parameter 'expected_ratio' has to be in range (0; 1>.
                    Received expected_ratio = {expected_ratio}.""")

            # 'offset_ratio' should not be greather than 'expected_ratio'.
            # expected ratio range = expected_ratio +/- offset_ratio.
            if offset_ratio < 0 or offset_ratio > 0.49:
                raise ValueError(
                    f"""Parameter 'offset_ratio' has to be in range <0; 0.49>.
                    Received offset_ratio = {offset_ratio}.""")

            if expected_ratio + offset_ratio > 1:
                raise ValueError(
                    f"""Ratio bigger than 100%.
                    Adjust 'expected_ratio' or 'offset_ratio' parameters.""")

            if expected_ratio - offset_ratio <= 0:
                raise ValueError(
                    f"""Ratio lower than or equal to 0%.
                    Adjust 'expected_ratio' or 'offset_ratio' parameters.""")


class RangeCriteriaException(Exception):
    """
    Raise exception, if user set 'expected_ratio'
    or 'offset_ratio' parameter, too low.
    """
