###########################################################################################
# HIP Hessian heads selected by `hessian_head_type`
#
# A self-contained Hessian prediction head that builds each off-diagonal 3x3 block
# as a *joint* nonlinear function of BOTH endpoint atoms plus the edge geometry,
# mirroring the successful HIP-EquiformerV2 design (concat source+target -> nonlinear
# message) in the e3nn/MACE framework.
#
# These heads are fully independent of the legacy and pair_mace_v1 code paths and
# of the older hessian_* off-diagonal option flags.
###########################################################################################

from typing import Any, Dict, List, Optional, Type, Union

import torch
from e3nn import nn, o3

from .blocks import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    RadialEmbeddingBlock,
)
from .hip import (
    add_hessian_graph_batch,
    blocks3x3_to_hessian,
    irreps_to_cartesian_matrix,
)
from .irreps_tools import tp_out_irreps_with_instructions
from .wrapper_ops import TensorProduct


class HIPHeadV2(torch.nn.Module):
    """Joint pair-conditioned HIP Hessian head (hessian_head_type='pair_v2').

    Diagonal blocks are predicted from per-node equivariant features.
    Off-diagonal blocks are predicted per directed edge i->j from a joint nonlinear
    function of the two endpoint node features and the edge geometry:

        concat(h_i, h_j) -> o3.Linear compress
            -> weighted TensorProduct with edge spherical harmonics (lmax=2)
            -> nn.Gate nonlinearity
            -> o3.Linear to (1x0e + 1x1e + 1x2e)
            -> Wigner-3j -> 3x3 Cartesian block

    The full Hessian is assembled with the shared `blocks3x3_to_hessian` helper, which
    adds each edge block at (i, j) and its transpose at (j, i), guaranteeing H = H^T.
    """

    def __init__(
        self,
        node_attr_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        hessian_hidden_irreps: Optional[o3.Irreps],
        hessian_feature_dim: int,
        interaction_cls: Type[InteractionBlock],
        num_interactions_hessian: int,
        correlation: Union[int, List[int]],
        num_elements: int,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        hessian_r_max: float,
        hessian_fully_connected: bool,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
        apply_cutoff: bool = True,
        hessian_radial_MLP: Optional[List[int]] = None,
        edge_irreps: Optional[o3.Irreps] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        use_reduced_cg: bool = True,
        use_agnostic_product: bool = False,
    ):
        super().__init__()

        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "hessian_r_max", torch.tensor(hessian_r_max, dtype=torch.get_default_dtype())
        )
        self.hessian_fully_connected = hessian_fully_connected
        self.num_interactions_hessian = num_interactions_hessian

        C = hessian_feature_dim
        self.hessian_feature_dim = hessian_feature_dim
        hessian_message_irreps = (hessian_hidden_irreps or hidden_irreps).simplify()
        self.hessian_message_irreps = hessian_message_irreps

        # Project backbone features into the head's working irrep space (if needed).
        if hessian_message_irreps == hidden_irreps.simplify():
            self.feature_proj = None
        else:
            self.feature_proj = o3.Linear(
                irreps_in=hidden_irreps, irreps_out=hessian_message_irreps
            )

        # ----- Edge geometry on the (fully-connected) Hessian graph -----
        sh_irreps_hessian = o3.Irreps.spherical_harmonics(lmax=2)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps_hessian, normalize=True, normalization="component"
        )
        if hessian_radial_MLP is None:
            hessian_radial_MLP = [64, 64, 64]
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=hessian_r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
        )
        radial_dim = self.radial_embedding.out_dim
        edge_feats_irreps = o3.Irreps(f"{radial_dim}x0e")

        # ----- Global-context refinement on the Hessian graph (EqV2 hessian_layers analog) -----
        # Correctly wired to lmax=2 edge spherical harmonics (unlike the legacy path,
        # which constructs these layers with the main-graph lmax SH but feeds lmax=2).
        num_features = hessian_message_irreps.count(o3.Irrep(0, 1))
        refine_target_irreps = (
            (sh_irreps_hessian * num_features).sort()[0].simplify()
        )
        corr = correlation[-1] if isinstance(correlation, (list, tuple)) else correlation
        self.refine_interactions = torch.nn.ModuleList()
        self.refine_products = torch.nn.ModuleList()
        for _ in range(num_interactions_hessian):
            interaction = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hessian_message_irreps,
                edge_attrs_irreps=sh_irreps_hessian,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=refine_target_irreps,
                hidden_irreps=hessian_message_irreps,
                avg_num_neighbors=avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=hessian_radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            product = EquivariantProductBasisBlock(
                node_feats_irreps=refine_target_irreps,
                target_irreps=hessian_message_irreps,
                correlation=corr,
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
            )
            self.refine_interactions.append(interaction)
            self.refine_products.append(product)

        # ----- Diagonal head (per node) -----
        self.proj_nodes = o3.Linear(
            hessian_message_irreps, o3.Irreps("1x0e + 1x1e + 1x2e")
        )

        # ----- Off-diagonal joint pair head (per directed edge) -----
        hessian_out_irreps = o3.Irreps(f"{C}x0e + {C}x1e + {C}x2e")
        self.hessian_out_irreps = hessian_out_irreps
        self.pair_context_irreps = (
            hessian_message_irreps
            + hessian_message_irreps
            + node_attr_irreps
            + node_attr_irreps
            + sh_irreps_hessian
            + edge_feats_irreps
        )

        # Compress the concatenated (h_i, h_j) pair to C channels per available irrep.
        pair_in_irreps = (hessian_message_irreps + hessian_message_irreps).simplify()
        unique_irreps = list(dict.fromkeys(ir for _, ir in hessian_message_irreps))
        compressed_irreps = o3.Irreps([(C, ir) for ir in unique_irreps]).sort()[0].simplify()
        self.pair_compress = o3.Linear(pair_in_irreps, compressed_irreps)

        # Joint bilinear coupling of the pair features with the edge direction.
        tp_mid_irreps, tp_instructions = tp_out_irreps_with_instructions(
            compressed_irreps,
            sh_irreps_hessian,
            hessian_out_irreps,
        )
        self.offdiag_tp = TensorProduct(
            compressed_irreps,
            sh_irreps_hessian,
            tp_mid_irreps,
            instructions=tp_instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        tp_weight_dim = radial_dim + 2 * node_attr_irreps.dim
        self.offdiag_tp_weights = nn.FullyConnectedNet(
            [tp_weight_dim] + hessian_radial_MLP + [self.offdiag_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Equivariant gate nonlinearity (e3nn analog of the EqV2 S2 activation).
        irreps_scalars = o3.Irreps(f"{C}x0e")
        irreps_gated = o3.Irreps(f"{C}x1e + {C}x2e")
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])
        self.gate = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.nn.functional.silu for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[torch.nn.functional.sigmoid for _ in irreps_gates],
            irreps_gated=irreps_gated,
        )
        self.pre_gate_linear = o3.Linear(tp_mid_irreps, self.gate.irreps_in)

        # Reduce to one channel per degree: 1x0e + 1x1e + 1x2e (9 components -> 3x3).
        self.proj_edges = o3.Linear(
            hessian_out_irreps, o3.Irreps("1x0e + 1x1e + 1x2e")
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        node_feats_list: List[torch.Tensor],
        edge_attrs: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
        envelope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Build the (fully-connected) Hessian graph and the scatter indices.
        data = add_hessian_graph_batch(
            data,
            hessian_r_max=self.hessian_r_max.item(),
            use_pbc=None,
            fully_connected=self.hessian_fully_connected,
        )
        edge_index = data["edge_index_hessian"]
        edge_distance = data["edge_distance_hessian"]
        edge_distance_vec = data["edge_distance_vec_hessian"]

        dtype = node_feats_list[0].dtype
        edge_attrs_hessian = self.spherical_harmonics(edge_distance_vec.to(dtype))

        edge_lengths = edge_distance
        if edge_lengths.dim() == 1:
            edge_lengths = edge_lengths.unsqueeze(-1)
        edge_feats_hessian, envelope_hessian = self.radial_embedding(
            edge_lengths,
            data["node_attrs"],
            edge_index,
            self.atomic_numbers,
        )

        # Backbone features -> working irrep space.
        node_feats = node_feats_list[-1]
        if self.feature_proj is not None:
            node_feats = self.feature_proj(node_feats)

        # Global-context refinement on the Hessian graph.
        for interaction, product in zip(
            self.refine_interactions, self.refine_products
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs_hessian,
                edge_feats=edge_feats_hessian,
                edge_index=edge_index,
                cutoff=envelope_hessian,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

        # Diagonal blocks (per node).
        diag_out = self.proj_nodes(node_feats)

        # Off-diagonal blocks (per directed edge), joint in both endpoints.
        sender = edge_index[0]
        receiver = edge_index[1]
        pair_feats = torch.cat(
            (node_feats[sender], node_feats[receiver]), dim=-1
        )
        pair_feats = self.pair_compress(pair_feats)

        tp_weight_inputs = torch.cat(
            (
                edge_feats_hessian,
                data["node_attrs"][sender],
                data["node_attrs"][receiver],
            ),
            dim=-1,
        )
        tp_weights = self.offdiag_tp_weights(tp_weight_inputs)
        if envelope_hessian is not None:
            tp_weights = tp_weights * envelope_hessian

        edge_hidden = self.offdiag_tp(pair_feats, edge_attrs_hessian, tp_weights)
        edge_hidden = self.pre_gate_linear(edge_hidden)
        edge_hidden = self.gate(edge_hidden)
        off_diag_out = self.proj_edges(edge_hidden)

        # (E, 3, 3) and (N, 3, 3) Cartesian blocks via Wigner-3j coupling.
        l012_edge_feat_3x3 = irreps_to_cartesian_matrix(off_diag_out)
        l012_node_feat_3x3 = irreps_to_cartesian_matrix(diag_out)

        return blocks3x3_to_hessian(
            edge_index=edge_index,
            data=data,
            l012_edge_features=l012_edge_feat_3x3,
            l012_node_features=l012_node_feat_3x3,
        )


class HIPHeadMessageV1(HIPHeadV2):
    """Message-centric HIP Hessian head (hessian_head_type='message_v1').

    This is the primary MACE-native head for testing the HIP-EquiformerV2 lesson:
    each off-diagonal block should be predicted from a rich directed pair message,
    not from a sender-only pre-aggregation convolution. It keeps the nonlinear
    pair tensor-product path from `HIPHeadV2` and adds a direct equivariant residual
    over the full pair context:

        h_i, h_j, z_i, z_j, Y_l(r_ij), radial(r_ij) -> 0e + 1e + 2e.

    The residual gives the head an explicit source-target message channel while
    preserving the same symmetric Hessian assembly as the other HIP heads.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_context_residual = o3.Linear(
            self.pair_context_irreps,
            self.hessian_out_irreps,
        )
        self.pair_context_residual_scale = torch.nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        node_feats_list: List[torch.Tensor],
        edge_attrs: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
        envelope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = add_hessian_graph_batch(
            data,
            hessian_r_max=self.hessian_r_max.item(),
            use_pbc=None,
            fully_connected=self.hessian_fully_connected,
        )
        edge_index = data["edge_index_hessian"]
        edge_distance = data["edge_distance_hessian"]
        edge_distance_vec = data["edge_distance_vec_hessian"]

        dtype = node_feats_list[0].dtype
        edge_attrs_hessian = self.spherical_harmonics(edge_distance_vec.to(dtype))

        edge_lengths = edge_distance
        if edge_lengths.dim() == 1:
            edge_lengths = edge_lengths.unsqueeze(-1)
        edge_feats_hessian, envelope_hessian = self.radial_embedding(
            edge_lengths,
            data["node_attrs"],
            edge_index,
            self.atomic_numbers,
        )

        node_feats = node_feats_list[-1]
        if self.feature_proj is not None:
            node_feats = self.feature_proj(node_feats)

        for interaction, product in zip(
            self.refine_interactions, self.refine_products
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs_hessian,
                edge_feats=edge_feats_hessian,
                edge_index=edge_index,
                cutoff=envelope_hessian,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

        diag_out = self.proj_nodes(node_feats)

        sender = edge_index[0]
        receiver = edge_index[1]
        pair_feats = torch.cat(
            (node_feats[sender], node_feats[receiver]), dim=-1
        )
        pair_feats = self.pair_compress(pair_feats)

        tp_weight_inputs = torch.cat(
            (
                edge_feats_hessian,
                data["node_attrs"][sender],
                data["node_attrs"][receiver],
            ),
            dim=-1,
        )
        tp_weights = self.offdiag_tp_weights(tp_weight_inputs)
        if envelope_hessian is not None:
            tp_weights = tp_weights * envelope_hessian

        edge_hidden = self.offdiag_tp(pair_feats, edge_attrs_hessian, tp_weights)
        edge_hidden = self.pre_gate_linear(edge_hidden)
        edge_hidden = self.gate(edge_hidden)

        pair_context = torch.cat(
            (
                node_feats[sender],
                node_feats[receiver],
                data["node_attrs"][sender],
                data["node_attrs"][receiver],
                edge_attrs_hessian,
                edge_feats_hessian,
            ),
            dim=-1,
        )
        edge_hidden = (
            edge_hidden
            + self.pair_context_residual_scale
            * self.pair_context_residual(pair_context)
        )

        off_diag_out = self.proj_edges(edge_hidden)

        l012_edge_feat_3x3 = irreps_to_cartesian_matrix(off_diag_out)
        l012_node_feat_3x3 = irreps_to_cartesian_matrix(diag_out)

        return blocks3x3_to_hessian(
            edge_index=edge_index,
            data=data,
            l012_edge_features=l012_edge_feat_3x3,
            l012_node_features=l012_node_feat_3x3,
        )


class HIPHeadEqV2(HIPHeadV2):
    """EquiformerV2-mirroring HIP Hessian head (hessian_head_type='eqv2_v1').

    Identical to :class:`HIPHeadV2` (joint two-body, gated off-diagonal message),
    but adds an explicit *parity-correct* branch to the off-diagonal block:

        (node 1o) x (edge 1o) -> (0e + 1e + 2e)

    A rank-2 Cartesian Hessian block is an even-parity object, so its vector part
    is `1e` (an even-parity / pseudovector), which equals the antisymmetric part of
    the off-diagonal block. In an O(3)-equivariant MACE backbone `1e` is not a native
    geometric feature, so the model must synthesise it from products of odd vectors.
    This branch supplies that `1e` content directly via `1o x 1o -> 1e`, which is the
    component that most strongly governs the soft (lowest) vibrational eigenvectors.

    Everything else - the diagonal head, the joint pair message, the gate, the graph,
    and the symmetric `blocks3x3_to_hessian` assembly - is inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        C = self.hessian_feature_dim
        radial_dim = self.radial_embedding.out_dim
        hessian_out_irreps = o3.Irreps(f"{C}x0e + {C}x1e + {C}x2e")

        # Project the (sender) node features onto C odd vectors for the parity branch.
        self.parity_node_vec = o3.Linear(
            self.hessian_message_irreps, o3.Irreps(f"{C}x1o")
        )
        # lmax=1 spherical harmonics of the edge direction; we slice out the 1o part.
        self.parity_spherical_harmonics_l1 = o3.SphericalHarmonics(
            o3.Irreps.spherical_harmonics(1),
            normalize=True,
            normalization="component",
        )
        # (C x 1o) x (1 x 1o) -> (C x 0e + C x 1e + C x 2e): even-parity rank-2 block.
        self.parity_tp = o3.FullyConnectedTensorProduct(
            o3.Irreps(f"{C}x1o"),
            o3.Irreps("1x1o"),
            hessian_out_irreps,
        )
        # Per-edge, per-channel scalar gate from the radial (and species) edge features.
        self.parity_gate = torch.nn.Linear(radial_dim, C)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        node_feats_list: List[torch.Tensor],
        edge_attrs: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
        envelope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Build the (fully-connected) Hessian graph and the scatter indices.
        data = add_hessian_graph_batch(
            data,
            hessian_r_max=self.hessian_r_max.item(),
            use_pbc=None,
            fully_connected=self.hessian_fully_connected,
        )
        edge_index = data["edge_index_hessian"]
        edge_distance = data["edge_distance_hessian"]
        edge_distance_vec = data["edge_distance_vec_hessian"]

        dtype = node_feats_list[0].dtype
        edge_attrs_hessian = self.spherical_harmonics(edge_distance_vec.to(dtype))

        edge_lengths = edge_distance
        if edge_lengths.dim() == 1:
            edge_lengths = edge_lengths.unsqueeze(-1)
        edge_feats_hessian, envelope_hessian = self.radial_embedding(
            edge_lengths,
            data["node_attrs"],
            edge_index,
            self.atomic_numbers,
        )

        # Backbone features -> working irrep space.
        node_feats = node_feats_list[-1]
        if self.feature_proj is not None:
            node_feats = self.feature_proj(node_feats)

        # Global-context refinement on the Hessian graph.
        for interaction, product in zip(
            self.refine_interactions, self.refine_products
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs_hessian,
                edge_feats=edge_feats_hessian,
                edge_index=edge_index,
                cutoff=envelope_hessian,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

        # Diagonal blocks (per node).
        diag_out = self.proj_nodes(node_feats)

        # Off-diagonal blocks (per directed edge), joint in both endpoints.
        sender = edge_index[0]
        receiver = edge_index[1]
        pair_feats = torch.cat(
            (node_feats[sender], node_feats[receiver]), dim=-1
        )
        pair_feats = self.pair_compress(pair_feats)

        tp_weight_inputs = torch.cat(
            (
                edge_feats_hessian,
                data["node_attrs"][sender],
                data["node_attrs"][receiver],
            ),
            dim=-1,
        )
        tp_weights = self.offdiag_tp_weights(tp_weight_inputs)
        if envelope_hessian is not None:
            tp_weights = tp_weights * envelope_hessian

        edge_hidden = self.offdiag_tp(pair_feats, edge_attrs_hessian, tp_weights)
        edge_hidden = self.pre_gate_linear(edge_hidden)
        edge_hidden = self.gate(edge_hidden)

        # Parity-correct (1o x 1o -> 0e + 1e + 2e) contribution.
        num_edges = edge_index.shape[1]
        C = self.hessian_feature_dim
        node_vec = self.parity_node_vec(node_feats)  # [N, 3C] = C x 1o
        node_vec_edge = node_vec[sender].view(num_edges, C, 3)
        gate = self.parity_gate(edge_feats_hessian)  # [E, C]
        if envelope_hessian is not None:
            gate = gate * envelope_hessian
        # Per-channel scalar scaling of an odd vector is equivariant.
        node_vec_edge = (node_vec_edge * gate.unsqueeze(-1)).reshape(num_edges, 3 * C)
        edge_l1 = self.parity_spherical_harmonics_l1(edge_distance_vec.to(dtype))
        edge_vec_l1 = edge_l1[:, 1:4]  # 1o part (drop the 0e component)
        parity_feats = self.parity_tp(node_vec_edge, edge_vec_l1)
        edge_hidden = edge_hidden + parity_feats

        off_diag_out = self.proj_edges(edge_hidden)

        # (E, 3, 3) and (N, 3, 3) Cartesian blocks via Wigner-3j coupling.
        l012_edge_feat_3x3 = irreps_to_cartesian_matrix(off_diag_out)
        l012_node_feat_3x3 = irreps_to_cartesian_matrix(diag_out)

        return blocks3x3_to_hessian(
            edge_index=edge_index,
            data=data,
            l012_edge_features=l012_edge_feat_3x3,
            l012_node_features=l012_node_feat_3x3,
        )
