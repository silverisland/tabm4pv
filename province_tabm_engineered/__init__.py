"""Engineered TabM interfaces for provincial ultra-short PV forecasting."""

__all__ = ["train", "test", "predict"]


def __getattr__(name):
    # Importing the deployment backbone must not import sklearn/joblib.
    if name in __all__:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)
