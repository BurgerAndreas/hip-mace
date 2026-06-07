from typing import List, Optional, Type

import torch
from e3nn import o3

from .blocks import EquivariantProductBasisBlock, InteractionBlock, RadialEmbeddingBlock
from .hip import (
    add_hessian_graph_batch,
    blocks3x3_to_hessian,
    enforce_hessian_translation_invariance,
    irreps_to_cartesian_matrix,
)
from .wrapper_ops import CuEquivarianceConfig, Linear, OEQConfig


def _interaction_irreps(
    sh_irreps: o3.Irreps,
    state_irreps: o3.Irreps,
    max_ell: int,
) -> o3.Irreps:
    """Match MACE's interaction irreps construction for a given hidden state."""
    num_features = state_irreps.count(o3.Irrep(0, 1))
    sh_irreps_inter = sh_irreps
    if state_irreps.count(o3.Irrep(0, -1)) > 0:
        sh_irreps_inter = o3.Irreps(
            "+".join([f"1x{ell}e+1x{ell}o" for ell in range(max_ell + 1)])
        )
    return (sh_irreps_inter * num_features).sort()[0].simplify()


class HIPGraphHessianHead(torch.nn.Module):
    """HIP-style Hessian readout for MACE features.

    This keeps HIP's invariant block assembly, but uses MACE interaction blocks
    on a dedicated Hessian graph to produce l<=2 edge and node irreps.
    """

    def __init__(
        self,
        *,
        node_attr_irreps: o3.Irreps,
        backbone_irreps: o3.Irreps,
        state_irreps: o3.Irreps,
        interaction_cls: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        correlation: int,
        atomic_numbers: List[int],
        avg_num_neighbors: float,
        hessian_r_max: float,
        hessian_edge_lmax: int,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: Optional[str],
        distance_transform: str,
        apply_cutoff: bool,
        radial_MLP: Optional[List[int]],
        hessian_feature_dim: int,
        edge_irreps: Optional[o3.Irreps] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
        use_reduced_cg: bool = True,
        use_agnostic_product: bool = False,
        fully_connected: bool = True,
        enforce_translation_invariance: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "hessian_r_max", torch.tensor(hessian_r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.fully_connected = fully_connected
        self.enforce_translation_invariance = enforce_translation_invariance

        self.input_proj = torch.nn.Identity()
        if backbone_irreps != state_irreps:
            self.input_proj = Linear(
                backbone_irreps,
                state_irreps,
                cueq_config=cueq_config,
            )

        hessian_sh_irreps = o3.Irreps.spherical_harmonics(hessian_edge_lmax)
        self.spherical_harmonics = o3.SphericalHarmonics(
            hessian_sh_irreps, normalize=True, normalization="component"
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=hessian_r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        hessian_interaction_irreps = _interaction_irreps(
            hessian_sh_irreps,
            state_irreps,
            hessian_edge_lmax,
        )
        self.interactions = torch.nn.ModuleList()
        self.products = torch.nn.ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(
                interaction_cls(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=state_irreps,
                    edge_attrs_irreps=hessian_sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=hessian_interaction_irreps,
                    hidden_irreps=state_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    edge_irreps=edge_irreps,
                    radial_MLP=radial_MLP,
                    cueq_config=cueq_config,
                    oeq_config=oeq_config,
                )
            )
            self.products.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=hessian_interaction_irreps,
                    target_irreps=state_irreps,
                    correlation=correlation,
                    num_elements=num_elements,
                    use_sc=True,
                    cueq_config=cueq_config,
                    oeq_config=oeq_config,
                    use_reduced_cg=use_reduced_cg,
                    use_agnostic_product=use_agnostic_product,
                )
            )

        self.edge_message = interaction_cls(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=state_irreps,
            edge_attrs_irreps=hessian_sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=state_irreps,
            hidden_irreps=state_irreps,
            avg_num_neighbors=avg_num_neighbors,
            edge_irreps=edge_irreps,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )

        head_irreps = o3.Irreps(
            f"{hessian_feature_dim}x0e + "
            f"{hessian_feature_dim}x1e + "
            f"{hessian_feature_dim}x2e"
        )
        block_irreps = o3.Irreps("1x0e + 1x1e + 1x2e")
        self.node_pre_readout = Linear(state_irreps, head_irreps, cueq_config=cueq_config)
        self.edge_pre_readout = Linear(state_irreps, head_irreps, cueq_config=cueq_config)
        self.node_readout = Linear(head_irreps, block_irreps, cueq_config=cueq_config)
        self.edge_readout = Linear(head_irreps, block_irreps, cueq_config=cueq_config)

    def _edge_features(self, data, dtype: torch.dtype):
        edge_index = data["edge_index_hessian"]
        edge_vec = data["edge_distance_vec_hessian"].to(dtype)
        edge_lengths = data["edge_distance_hessian"]
        if edge_lengths.dim() == 1:
            edge_lengths = edge_lengths.unsqueeze(-1)

        edge_attrs = self.spherical_harmonics(edge_vec)
        edge_feats, envelope = self.radial_embedding(
            edge_lengths,
            data["node_attrs"],
            edge_index,
            self.atomic_numbers,
        )
        return edge_index, edge_attrs, edge_feats, envelope

    def forward(self, data, node_feats: torch.Tensor) -> torch.Tensor:
        data = add_hessian_graph_batch(
            data,
            hessian_r_max=float(self.hessian_r_max.item()),
            use_pbc=None,
            fully_connected=self.fully_connected,
        )
        edge_index, edge_attrs, edge_feats, envelope = self._edge_features(
            data, node_feats.dtype
        )

        node_feats = self.input_proj(node_feats)
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=envelope,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

        raw_messages = self.edge_message(
            node_attrs=data["node_attrs"],
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=envelope,
            return_raw_messages=True,
        )
        edge_state = self.edge_message.linear(raw_messages)
        edge_state = edge_state / self.edge_message.avg_num_neighbors

        edge_irreps = self.edge_readout(self.edge_pre_readout(edge_state))
        node_irreps = self.node_readout(self.node_pre_readout(node_feats))
        edge_blocks = irreps_to_cartesian_matrix(edge_irreps)
        node_blocks = irreps_to_cartesian_matrix(node_irreps)

        hessian = blocks3x3_to_hessian(
            edge_index=edge_index,
            data=data,
            l012_edge_features=edge_blocks,
            l012_node_features=node_blocks,
        )
        if self.enforce_translation_invariance:
            hessian = enforce_hessian_translation_invariance(hessian, data["natoms"])
        return hessian
