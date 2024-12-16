from abc import ABC, abstractmethod
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
numpy_to_torch_dtype_dict = {
    np.dtype('bool'): torch.bool,
    np.dtype('uint8'): torch.uint8,
    np.dtype('int8'): torch.int8,
    np.dtype('int16'): torch.int16,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('complex64'): torch.complex64,
    np.dtype('complex128'): torch.complex128
}

# TODO is it better to write a meta class that takes many schedulers/optimizers etc
# class CosineAnnelaingDual(torch.optim.lr_scheduler.CosineAnnealingLR):

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}

def tensor2numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x

def cast_tensor_to_numpy(function, x, areturn=True, **kwargs):
    if not torch.is_tensor(x):
        x = tensor2numpy(x)
        dtype = numpy_to_torch_dtype_dict[x.dtype]
    else:
        dtype = x.dtype
    x = function(x, **kwargs)
    if areturn:
        x = torch.tensor(x, dtype=dtype)
        return x


class BasePreprocess(ABC):

    @abstractmethod
    def fit(self, X, **kwargs):
        return torch.tensor(0)

    @abstractmethod
    def transform(self, X, **kwargs):
        return 0

    @abstractmethod
    def inverse_transform(self, X, **kwargs):
        return 0

    def fit_transform(self, X, **kwargs):
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)


class Standardiser(StandardScaler):

    def fit(self, X, **kwargs):
        cast_tensor_to_numpy(super(Standardiser, self).fit, X, areturn=False, **kwargs)

    def transform(self, X, **kwargs):
        return cast_tensor_to_numpy(super(Standardiser, self).transform, X, **kwargs)

    def inverse_transform(self, X, **kwargs):
        return cast_tensor_to_numpy(super(Standardiser, self).inverse_transform, X, **kwargs)


class BaseMassSeparate(BasePreprocess):

    @abstractmethod
    def _fit(self, data, **kwargs):
        return 0

    @abstractmethod
    def _forward_method(self, data, info, **kwargs):
        return 0

    @abstractmethod
    def _inverse_method(self, data, info, **kwargs):
        return 0

    def preprocessed(self):
        if not hasattr(self, 'info'):
            raise Exception('Scaler must be fit first')

    def fit(self, data, **kwargs):
        self.n_features = data.shape[1]
        self.info = self._fit(data, **kwargs)

    def transform(self, data, **kwargs):
        self.preprocessed()
        X = self._forward_method(data, self.info.to(data), **kwargs)
        return X

    def inverse_transform(self, data, **kwargs):
        self.preprocessed()
        return self._inverse_method(data, self.info.to(data), **kwargs)


class Normaliser(BaseMassSeparate):

    def __init__(self):
        self.eps = 1e-6
        self.rel_eps = 0.001

    def _fit(self, data, **kwargs):
        mx = data.max(0)[0]
        mn = data.min(0)[0]
        return torch.hstack([mx + abs(mx * self.rel_eps + self.eps), mn - abs(mn * self.rel_eps + self.eps)])

    def _forward_method(self, data, info, **kwargs):
        mx = info[:self.n_features]
        mn = info[self.n_features:]
        data = (data - mn) / (mx - mn)
        return data

    def _inverse_method(self, data, info, **kwargs):
        mx = info[:self.n_features]
        mn = info[self.n_features:]
        data = data * (mx - mn) + mn
        return data


class QuantileScaler(Normaliser):

    def __init__(self, q1=0.01, q2=0.99):
        super(QuantileScaler, self).__init__()
        self.q1 = q1
        self.q2 = q2

    def _fit(self, data, **kwargs):
        return torch.cat([data.quantile(self.q2, 0), data.quantile(self.q1, 0)])


class LogitScale(BaseMassSeparate):

    def __init__(self):
        self.eps = 1e-3

    def _fit(self, data, **kwargs):
        # As the data is to be log scaled we will need to clamp some values
        return torch.tensor([data.min(), data.max()])

    def _forward_method(self, data, info, **kwargs):
        data = data.clamp(*info)
        data = data.log() - (1 - data).log()
        return data

    def _inverse_method(self, data, info, **kwargs):
        data = data.exp() / (data.exp() + 1)
        # The only issue you get here is when .exp returns inf, which will result in an inf here that can be reset
        data[data.isnan()] = 1.
        return data


class ReCenter(BaseMassSeparate):

    def __init__(self, scale=1):
        self.scale = scale

    def _fit(self, data, **kwargs):
        return torch.tensor(0)

    def _forward_method(self, data, info, **kwargs):
        return (data - 0.5) * 2 * self.scale

    def _inverse_method(self, data, info, **kwargs):
        return data / (2 * self.scale) + 0.5


class CompositePreprocess(BaseMassSeparate):

    def __init__(self, transformer_list, *args, **kwargs):
        self.transfomer_list = transformer_list

    def _fit(self, data, **kwargs):
        for transformer in self.transfomer_list:
            data = transformer.fit_transform(data)
        return torch.tensor(0)

    def _forward_method(self, data, info, **kwargs):
        for transformer in self.transfomer_list:
            data = transformer.transform(data)
        return data

    def _inverse_method(self, data, info, **kwargs):
        for transformer in self.transfomer_list[::-1]:
            data = transformer.inverse_transform(data)
        return data


class CathodePreprocess(BaseMassSeparate):

    def __init__(self):
        self.features_preprocess = CompositePreprocess(
            [Normaliser(),
             LogitScale(),
             QuantileScaler()]
        )
        self.mass_transformer = Normaliser()
        self.final_transform = ReCenter(2)

    def _fit(self, data, **kwargs):
        rest_data, mass = data[:, :-1], data[:, -1:]
        rest_data = self.features_preprocess.fit_transform(rest_data)
        rest_mass = self.mass_transformer.fit_transform(mass)
        rest_data = torch.cat((rest_data, rest_mass), 1)
        self.final_transform.fit(rest_data)
        return torch.tensor(0)

    def _forward_method(self, data, info, **kwargs):

        if len(data.shape) > 1:
            rest_data, mass = data[:, :-1], data[:, -1:]
            rest_data = self.features_preprocess.transform(rest_data)
            mass = self.mass_transformer.transform(mass)
            rest_data = torch.cat((rest_data, mass), 1)
            rest_data = self.final_transform.transform(rest_data)
            return rest_data
        else:
            rest_data = self.mass_transformer.transform(rest_data)
            rest_data = self.final_transform.transform(rest_data)
            return rest_data

    def _inverse_method(self, data, info, **kwargs):

        data = self.final_transform.inverse_transform(data)

        if len(data.shape) > 1:
            data, mass = data[:, :-1], data[:, -1:]
            mass = self.mass_transformer.inverse_transform(mass)
            data = self.features_preprocess.inverse_transform(data)
            data = torch.cat((data, mass), 1)
            return data
        else:
            return self.mass_transformer.inverse_transform(data)

    def prep_mass(self, mass):
        data = self.mass_transformer.transform(mass)
        data = self.final_transform.transform(data)
        return data

    def inv_mass(self, mass):
        data = self.final_transform.inverse_transform(mass)
        return self.mass_transformer.inverse_transform(data)
