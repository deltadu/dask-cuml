#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import cudf
import dask
import dask_cudf

import nvcategory
from librmm_cffi import librmm


def _enforce_str(y):
    if y.dtype != "object":
        return y.astype("str")
    return y


def _trans(ser, categories):
    encoded = (nvcategory
               .from_strings(ser.data)
               .set_keys(categories.keys()))
    device_array = librmm.device_array(len(ser.data), dtype=np.int32)
    encoded.values(devptr=device_array.device_ctypes_pointer.value)
    ser = cudf.Series(device_array)
    return ser


# def _trans(ser, categories):
#     encoded = (nvcategory
#                .from_strings(ser.data)
#                .set_keys(categories.keys())
#                .values)
#     return encoded


class LabelEncoder(object):

    def __init__(self, *args, **kwargs):
        self._cats = None
        self._dtype = None
        self._fitted = False


    def _check_is_fitted(self):
        if (not self._fitted) or (self._cats is None):
            raise ValueError('LabelEncoder must be fit first')


    def fit(self, y):
        if isinstance(y, dask_cudf.Series):
            y = y.map_partitions(_enforce_str)
            self._cats = nvcategory.from_strings(y.unique().compute().data)
        elif isinstance(y, cudf.Series):
            y = _enforce_str(y)
            self._cats = nvcategory.from_strings(y.data)
        else:
            raise TypeError('Input of type {} is not dask_cudf.Series '
                            + 'or cudf.Series'.format(type(y)))
        self._fitted = True
        self._dtyp = y.dtype

        return self


    def transform(self, y):
        self._check_is_fitted()

        if isinstance(y, dask_cudf.Series):
            y = y.map_partitions(_enforce_str)
            encoded = y.map_partitions(_trans, self._cats)
            if len(encoded[encoded == -1].compute()) != 0:
                raise ValueError('contains previously unseen labels')

        elif isinstance(y, cudf.Series):
            y = _enforce_str(y)
            encoded = _trans(y, self._cats)
            if -1 in encoded:
                raise ValueError('contains previously unseen labels')

        else:
            raise TypeError(
                'Input of type {} is not dask_cudf.Series '
                + 'or cudf.Series'.format(type(y)))
        return encoded


    def fit_transform(self, y):
        if isinstance(y, dask_cudf.Series):
            y = y.map_partitions(_enforce_str)
            self._cats = nvcategory.from_strings(y.unique().compute().data)
            self._fitted = True

            encoded = y.map_partitions(_trans, self._cats)
            if len(encoded[encoded == -1].compute()) != 0:
                raise ValueError('contains previously unseen labels')

        elif isinstance(y, cudf.Series):
            y = _enforce_str(y)
            self._cats = nvcategory.from_strings(y.data)
            self._fitted = True

            encoded = _trans(y, self._cats)
            if -1 in encoded:
                raise ValueError('contains previously unseen labels')
        else:
            raise TypeError(
                'Input of type {} is not dask_cudf.Series '
                + 'or cudf.Series'.format(type(y)))
        
        self._dtype = y.dtype
        return encoded


    def inverse_transform(self, y):
        raise NotImplementedError
