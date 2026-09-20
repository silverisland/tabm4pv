"""Shared TabM construction for training and standalone inference."""

import rtdl_num_embeddings
import tabm
import torch


def make_model(
    n_features: int, device: torch.device, architecture: dict | None = None
) -> torch.nn.Module:
    embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_features)
    return tabm.TabM.make(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_out=1,
        num_embeddings=embeddings,
        **(architecture or {}),
    ).to(device)
