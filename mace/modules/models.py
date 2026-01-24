###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.embeddings import GenericJointEmbedding
from mace.modules.radial import ZBLBasis
from mace.tools.scatter import scatter_mean, scatter_sum
from mace.tools.torch_tools import get_change_of_basis, spherical_to_cartesian

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    # RealAgnosticResidualInteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipolePolarReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from .utils import (
    compute_dielectric_gradients,
    compute_fixed_charge_dipole,
    compute_fixed_charge_dipole_polar,
    get_atomic_virials_stresses,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
    prepare_graph,
)
from .hip import add_hessian_graph_batch, blocks3x3_to_hessian, irreps_to_cartesian_matrix


@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        pair_repulsion: bool = False,
        apply_cutoff: bool = True,
        use_reduced_cg: bool = True,
        use_so3: bool = False,
        use_agnostic_product: bool = False,
        use_last_readout_only: bool = False,
        use_embedding_readout: bool = False,
        distance_transform: str = "None",
        edge_irreps: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        heads: Optional[List[str]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,
        embedding_specs: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
        lammps_mliap: Optional[bool] = False,
        readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
        # Added for HIP Hessian prediction
        hip: bool = False,
        hessian_feature_dim: int = 32,
        hessian_use_last_layer_only: bool = False,
        hessian_r_max: float = 16.0,
        hessian_edge_lmax: int = 3, # 2 or 3
        hessian_use_radial: bool = True,  # Use radial embeddings
        hessian_use_both_nodes: bool = True,  # Use both h_i and h_j (False = only h_j)
        hessian_aggregation: str = "learnable",  # "mean", "learnable"
        hessian_edge_feature_method: str = "message_passing",  # "edge_tp" or "message_passing"
        hessian_message_passing_layer: Optional[int] = None,  # Which interaction layer to use (None = last)
        hessian_use_directional_encoding: bool = False,  # Include normalized edge vector r_ij in tensor product
        hessian_separate_radial_network: bool = False,  # Use dedicated radial MLP for Hessian (not shared with energy)
        hessian_radial_MLP: Optional[List[int]] = None,  # Radial MLP architecture for separate network
        hessian_use_edge_gates: bool = False,  # Add equivariant gating on off-diagonal features
        num_interactions_hessian: int = 0,  # Number of additional interaction layers for Hessian prediction
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if heads is None:
            heads = ["Default"]
        self.heads = heads
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        self.lammps_mliap = lammps_mliap
        self.apply_cutoff = apply_cutoff
        self.edge_irreps = edge_irreps
        self.use_reduced_cg = use_reduced_cg
        self.use_agnostic_product = use_agnostic_product
        self.use_so3 = use_so3
        self.use_last_readout_only = use_last_readout_only

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        # adds up scalar irreps with even parity
        embedding_size = node_feats_irreps.count(o3.Irrep(0, 1))
        if embedding_specs is not None:
            self.embedding_specs = embedding_specs
            self.joint_embedding = GenericJointEmbedding(
                base_dim=embedding_size,
                embedding_specs=embedding_specs,
                out_dim=embedding_size,
            )
            if use_embedding_readout:
                self.embedding_readout = LinearReadoutBlock(
                    node_feats_irreps,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")
        if pair_repulsion:
            self.pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)
            self.pair_repulsion = True

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))

        # interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l + 1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
            use_reduced_cg=use_reduced_cg,
            use_agnostic_product=use_agnostic_product,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        if not use_last_readout_only:
            self.readouts.append(
                LinearReadoutBlock(
                    hidden_irreps,
                    o3.Irreps(f"{len(heads)}x0e"),
                    cueq_config,
                    oeq_config,
                )
            )

        for i in range(num_interactions - 1):
            if i == (num_interactions - 2) and not hip:
                # Select only scalars for last layer
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
                oeq_config=oeq_config,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    readout_cls(
                        hidden_irreps_out,
                        (len(heads) * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps(f"{len(heads)}x0e"),
                        len(heads),
                        cueq_config,
                        oeq_config,
                    )
                )
            elif not use_last_readout_only:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        o3.Irreps(f"{len(heads)}x0e"),
                        cueq_config,
                        oeq_config,
                    )
                )

        self.hessian_use_last_layer_only = hessian_use_last_layer_only
        self.hip = hip
        self.register_buffer(
            "hessian_r_max", torch.tensor(hessian_r_max, dtype=torch.get_default_dtype())
        )
        if self.hip:
            # consider adding one full layer here
            # hessian_hidden_irreps = o3.Irreps("64x0e + 64x1o + 64x2e")
            # self.hessian_interaction = interaction_cls(
            #     node_attrs_irreps=node_attr_irreps,
            #     node_feats_irreps=hidden_irreps,
            #     edge_attrs_irreps=sh_irreps,
            #     edge_feats_irreps=edge_feats_irreps,
            #     target_irreps=interaction_irreps,
            #     hidden_irreps=hessian_hidden_irreps,
            #     avg_num_neighbors=avg_num_neighbors,
            #     edge_irreps=edge_irreps,
            #     radial_MLP=radial_MLP,
            #     cueq_config=cueq_config,
            #     oeq_config=oeq_config,
            # )
            # prod = EquivariantProductBasisBlock(
            #     node_feats_irreps=interaction_irreps,
            #     target_irreps=hessian_hidden_irreps, 
            #     # ...
            # )
            
            assert hidden_irreps.count(o3.Irrep(2, 1)) > 0, \
                f"Hessian requires 2e in hidden_irreps but got {hidden_irreps}. Try adding e.g. '+ 64x2e' to hidden_irreps."
            # The Hessian is an Even Parity object, 
            # so we require "Cx0e + Cx1e + Cx2e" instead of the
            # "Cx1o" used for forces. Luckily, the Tensor Product of two 
            # Odd vectors (node feature $1o$ $\otimes$ edge attribute $1o$) 
            # produces the required $0e, 1e, 2e$ output.
            # hidden_irreps: The configuration of your main backbone (e.g., "128x0e + 32x1o")
            # hessian_feature_dim: How many channels you want per L component (0,1,2)
            
            # Output Definition: 0e, 1e, 2e (All Even for Hessian)
            hessian_out_irreps = o3.Irreps(f"{hessian_feature_dim}x0e + {hessian_feature_dim}x1e + {hessian_feature_dim}x2e")
            
            # Node Projector (for Diagonal)
            # We assume node_feats already contains mixed parities ($0e, 1e or 1o, 2e$)
            self.hessian_proj_nodes_layerwise = o3.Linear(
                irreps_in=hidden_irreps, irreps_out=hessian_out_irreps
            )

            # Edge Extractor (for Off-Diagonal)
            # Input 1: Node features of neighbor j (hidden_irreps)
            # Input 2: Edge Geometry (Spherical Harmonics usually up to L=2 or 3)
            # We assume edge_attrs are standard Spherical Harmonics (0e + 1o + 2e + ...)
            # lmax=2 is sufficient, but lmax=3 adds a parity-correct path $1o \otimes 3o \to 2e$
            sh_irreps_hessian = o3.Irreps.spherical_harmonics(lmax=hessian_edge_lmax) 
            
            self.hessian_spherical_harmonics = o3.SphericalHarmonics(
                sh_irreps_hessian, normalize=True, normalization="component"
            )
            
            # Store flags for feature computation
            self.hessian_use_radial = hessian_use_radial
            self.hessian_use_both_nodes = hessian_use_both_nodes
            self.hessian_aggregation = hessian_aggregation
            self.hessian_edge_feature_method = hessian_edge_feature_method
            self.hessian_message_passing_layer = hessian_message_passing_layer
            self.hessian_use_directional_encoding = hessian_use_directional_encoding
            self.hessian_separate_radial_network = hessian_separate_radial_network
            self.hessian_use_edge_gates = hessian_use_edge_gates
            
            # Validate hessian_edge_feature_method
            assert hessian_edge_feature_method in ["edge_tp", "message_passing"], \
                f"hessian_edge_feature_method must be 'edge_tp' or 'message_passing', got {hessian_edge_feature_method}"
            
            # Separate radial network for Hessian (if enabled)
            if hessian_separate_radial_network:
                if hessian_radial_MLP is None:
                    hessian_radial_MLP = [64, 64, 64]
                self.hessian_radial_embedding = RadialEmbeddingBlock(
                    r_max=hessian_r_max,
                    num_bessel=num_bessel,
                    num_polynomial_cutoff=num_polynomial_cutoff,
                    radial_type=radial_type,
                    distance_transform=distance_transform,
                    apply_cutoff=apply_cutoff,
                )
                hessian_radial_dim = self.hessian_radial_embedding.out_dim
            else:
                hessian_radial_dim = self.radial_embedding.out_dim
            
            # If using both nodes, need to combine h_i and h_j
            # When False, we only use h_j (drop h_i entirely)
            if hessian_use_both_nodes:
                # Combine two node features: h_i and h_j -> combined features
                # Options: concatenate and project, or add and project
                # Using addition + projection for simplicity and equivariance
                self.hessian_node_combine = o3.Linear(
                    irreps_in=hidden_irreps,
                    irreps_out=hidden_irreps
                )
            
            # If using radial features, need to incorporate them
            if hessian_use_radial:
                # Radial features are scalar (0e), so we can use them to gate/modulate
                # the edge features. Create a projection from radial features to modulate
                # the tensor product output.
                radial_irreps = o3.Irreps(f"{hessian_radial_dim}x0e")
                # Project radial features to match hessian_out_irreps for gating
                self.hessian_radial_proj = o3.Linear(
                    irreps_in=radial_irreps,
                    irreps_out=hessian_out_irreps
                )
            
            # Tensor product: node features x spherical harmonics -> hessian features
            # If using directional encoding, we'll need to handle r_ij separately
            if hessian_use_directional_encoding:
                # For directional encoding: TP(h_j, Y_ij, r_ij) where r_ij is normalized vector
                # Convert normalized Cartesian vector r_ij to spherical harmonics (1o)
                # Then combine Y_ij and r_ij first, then with h_j
                r_ij_sh_irreps = o3.Irreps("1x1o")  # Normalized direction vector as spherical harmonics
                self.hessian_r_ij_spherical_harmonics = o3.SphericalHarmonics(
                    r_ij_sh_irreps, normalize=True, normalization="component"
                )
                # Combine Y_ij and r_ij first, then with h_j
                combined_edge_irreps = (sh_irreps_hessian * r_ij_sh_irreps).sort()[0].simplify()
                self.edge_tp_directional = o3.FullyConnectedTensorProduct(
                    irreps_in1=sh_irreps_hessian,
                    irreps_in2=r_ij_sh_irreps,
                    irreps_out=combined_edge_irreps
                )
                self.edge_tp = o3.FullyConnectedTensorProduct(
                    irreps_in1=hidden_irreps,
                    irreps_in2=combined_edge_irreps,
                    irreps_out=hessian_out_irreps
                )
            else:
                self.edge_tp = o3.FullyConnectedTensorProduct(
                    irreps_in1=hidden_irreps,
                    irreps_in2=sh_irreps_hessian, 
                    irreps_out=hessian_out_irreps
                )
            
            # Edge-level gates for non-linearity on off-diagonal features
            if hessian_use_edge_gates:
                # Create equivariant gating on off-diagonal features
                irreps_scalars = o3.Irreps(
                    [(mul, ir) for mul, ir in hessian_out_irreps if ir.l == 0]
                )
                irreps_gated = o3.Irreps([(mul, ir) for mul, ir in hessian_out_irreps if ir.l > 0])
                irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])
                self.hessian_edge_gate = nn.Gate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[torch.nn.functional.silu for _ in irreps_scalars],
                    irreps_gates=irreps_gates,
                    act_gates=[torch.nn.functional.sigmoid] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
            
            # Learnable layer aggregation weights (if using learnable aggregation)
            if hessian_aggregation == "learnable":
                # Learnable logits for softmax weighting across layers
                # Initialize with zeros so initial weights are uniform (softmax(0) = 1/n)
                # Size accounts for main backbone layers + additional Hessian layers
                max_layers = num_interactions + num_interactions_hessian
                self.hessian_layer_weights_param = torch.nn.Parameter(
                    torch.zeros(max_layers)
                )
            
            # Output: One channel per degree (1+3+5 = 9 total elements)
            # All Even parity (e) for Hessian
            # The Linear layer reduces multiplicity hessian_out_irreps -> 1
            self.hessian_proj_nodes = o3.Linear(hessian_out_irreps, o3.Irreps("1x0e + 1x1e + 1x2e"))
            self.hessian_proj_edges = o3.Linear(hessian_out_irreps, o3.Irreps("1x0e + 1x1e + 1x2e"))
            
            # If using message passing, create a dedicated interaction block for Hessian edge features
            if hessian_edge_feature_method == "message_passing":
                # Create a specialized interaction block for Hessian edge features
                # Use the same interaction class as the main model
                # Configure it specifically for Hessian edge feature extraction
                hessian_edge_feats_irreps = o3.Irreps(f"{hessian_radial_dim}x0e")
                
                # # Compute interaction irreps for Hessian: sh_irreps * hidden_irreps
                # num_features = hidden_irreps.count(o3.Irrep(0, 1))
                # hessian_interaction_irreps = (sh_irreps_hessian * num_features).sort()[0].simplify()
                
                # Use separate radial MLP if specified
                hessian_radial_MLP_for_interaction = hessian_radial_MLP if hessian_separate_radial_network else radial_MLP
                
                # Create the Hessian-specific interaction block
                # The target_irreps should match what we want to project to hessian_out_irreps
                # Use hidden_irreps as target to maintain compatibility with the projection layer
                print(f"Interaction block: {interaction_cls.__name__}")
                self.hessian_interaction = interaction_cls(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=sh_irreps_hessian,
                    edge_feats_irreps=hessian_edge_feats_irreps,
                    target_irreps=hidden_irreps,  # Changed from hessian_interaction_irreps to hidden_irreps
                    hidden_irreps=hidden_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    edge_irreps=edge_irreps,
                    radial_MLP=hessian_radial_MLP_for_interaction,
                    cueq_config=cueq_config,
                    oeq_config=oeq_config,
                )
                
                # Create projection layer from hidden_irreps to hessian_out_irreps
                self.hessian_message_proj = o3.Linear(
                    irreps_in=hidden_irreps,  # Changed from hessian_interaction_irreps to hidden_irreps
                    irreps_out=hessian_out_irreps
                )

            # Additional interaction layers for Hessian (if num_interactions_hessian > 0)
            # These layers use the main graph to further refine node features before Hessian extraction
            self.num_interactions_hessian = num_interactions_hessian
            if num_interactions_hessian > 0:
                self.hessian_interactions = torch.nn.ModuleList()
                self.hessian_products = torch.nn.ModuleList()

                # Create num_interactions_hessian additional layers using main graph architecture
                for i in range(num_interactions_hessian):
                    # These layers process on the main graph, so use main graph parameters
                    hessian_inter = interaction_cls(
                        node_attrs_irreps=node_attr_irreps,
                        node_feats_irreps=hidden_irreps,
                        edge_attrs_irreps=sh_irreps,  # Use main graph spherical harmonics
                        edge_feats_irreps=edge_feats_irreps,  # Use main graph edge features
                        target_irreps=interaction_irreps,
                        hidden_irreps=hidden_irreps,
                        avg_num_neighbors=avg_num_neighbors,
                        edge_irreps=edge_irreps,
                        radial_MLP=radial_MLP,  # Use main radial MLP
                        cueq_config=cueq_config,
                        oeq_config=oeq_config,
                    )
                    self.hessian_interactions.append(hessian_inter)

                    hessian_prod = EquivariantProductBasisBlock(
                        node_feats_irreps=interaction_irreps,
                        target_irreps=hidden_irreps,
                        correlation=correlation[-1] if isinstance(correlation, list) else correlation,
                        num_elements=num_elements,
                        use_sc=True,
                        cueq_config=cueq_config,
                        oeq_config=oeq_config,
                        use_reduced_cg=use_reduced_cg,
                        use_agnostic_product=use_agnostic_product,
                    )
                    self.hessian_products.append(hessian_prod)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        # added for HIP Hessian prediction
        predict_hessian: bool = False, 
    ) -> Dict[str, Optional[torch.Tensor]]:
        """ """
        # L = sum_{i=0}^l (2l+1) = 1 + 3 + 5 + 7 + 9 + ...
            
        # Setup
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, n_heads]
        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_node_energy = torch.zeros_like(node_e0)
            pair_energy = torch.zeros_like(e0)

        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        energies = [e0, pair_energy]
        node_energies_list = [node_e0, pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            # [BN, hidden_irreps, L], [BN, interaction_irreps]
            # C = hidden_irreps
            # interaction_irreps = 1*C + 3*C + 5*C + ... + (2l+1)*C
            # sc = Self-Connection / Atomic Basis A 
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice, # [BN, atomic_numers]
                node_feats=node_feats, # [BN, hidden_irreps]
                edge_attrs=edge_attrs, # [E, L]
                edge_feats=edge_feats, # [E, Bessel]
                edge_index=data["edge_index"], # [2, E]
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            # [BN, hidden_irreps]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)
        
        # Mace typically has one energy readout per layer
        # which are summed up
        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ] # [BN]
            energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
            energies.append(energy) # [B]
            node_energies_list.append(node_es)

        contributions = torch.stack(energies, dim=-1) # [B, num_readouts]
        total_energy = torch.sum(contributions, dim=-1) # [B]
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1) # [BN]
        # node_feats_out = torch.cat(node_feats_list, dim=-1) # [BN, hidden_irreps*num_interactions]

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )

        if predict_hessian:
            hessian = self._predict_hessian_hip(
                data, node_feats_list, edge_attrs, edge_feats, cutoff
            )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            # Hessian prediction will overwrite the autodiff Hessian
            "hessian": hessian,
        }
    
    def _extract_raw_messages_from_interaction(
        self,
        interaction: InteractionBlock,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract per-edge messages from an interaction block before aggregation.
        
        Returns:
            mji: [n_edges, irreps] - raw messages per edge
        """
        result = interaction(
            node_attrs=node_attrs,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
            return_raw_messages=True,
        )
        if isinstance(result, tuple):
            return result[0]  # Return the messages, not the sc
        return result
        
    
    def _predict_hessian_hip(
        self,
        data,
        node_feats_list,
        edge_attrs,
        edge_feats,
        cutoff,
    ) -> torch.Tensor:
        """Predict Hessian from l=0,1,2 features."""
        # Compute the graph for the Hessian
        # Usually the graph is computed in AtomicData.__init__()
        data = add_hessian_graph_batch(
            data,
            cutoff=self.hessian_r_max.item(),
            # max_neighbors=self.max_neighbors,
            # use_pbc=data["pbc"][-1] if len(data["pbc"]) > 3 else data["pbc"],
            use_pbc=None,
        )
        # if otf_graph or not hasattr(data, "nedges_hessian"):
        # else:
        #     data = add_extra_props_for_hessian(data)
        
        # For the diagonal elements of the Hessian,
        # we need node features of size [BN, L].
        # We could get node features for example by:
        # (1) reading out each layer's features similar to the energy,
        # or aggregate by e.g. averaging,
        # (2) only using the last layer's features,
        # We could use either sc or node_feats after the product.
        # Similarly, there are many ways to get off-diagonal features.
        
        edge_index_hessian = data["edge_index_hessian"]
        edge_distance_hessian = data["edge_distance_hessian"]
        edge_distance_vec_hessian = data["edge_distance_vec_hessian"]
        
        edge_attrs_hessian = self.hessian_spherical_harmonics(
            edge_distance_vec_hessian.to(node_feats_list[0].dtype)
        )
        
        # Compute radial embeddings for Hessian edges (if enabled)
        edge_feats_hessian = None
        if self.hessian_use_radial:
            # Ensure edge_distance_hessian is [E, 1] for radial_embedding
            edge_lengths_hessian = edge_distance_hessian
            if edge_lengths_hessian.dim() == 1:
                edge_lengths_hessian = edge_lengths_hessian.unsqueeze(-1)
            # Use separate radial network if enabled, otherwise use shared one
            if self.hessian_separate_radial_network:
                edge_feats_hessian, _ = self.hessian_radial_embedding(
                    edge_lengths_hessian,
                    data["node_attrs"],
                    edge_index_hessian,
                    self.atomic_numbers
                )
            else:
                edge_feats_hessian, _ = self.radial_embedding(
                    edge_lengths_hessian,
                    data["node_attrs"],
                    edge_index_hessian,
                    self.atomic_numbers
                )
        
        # Normalized edge vectors for directional encoding (if enabled)
        edge_vec_normalized_sh = None
        if self.hessian_use_directional_encoding:
            # Normalize edge_distance_vec_hessian to get r_ij
            edge_vec_norm = torch.norm(edge_distance_vec_hessian, dim=-1, keepdim=True)
            edge_vec_normalized = edge_distance_vec_hessian / (edge_vec_norm + 1e-8)
            # Convert normalized Cartesian vector to spherical harmonics (1o)
            edge_vec_normalized_sh = self.hessian_r_ij_spherical_harmonics(
                edge_vec_normalized.to(node_feats_list[0].dtype)
            )

        # Run additional Hessian-specific interaction layers if configured
        # These layers use the main graph to further refine node features
        if self.num_interactions_hessian > 0:
            # Start with the last node features from the main backbone
            node_feats_hessian = node_feats_list[-1]

            # Run Hessian-specific interaction layers on the main graph
            for interaction, product in zip(self.hessian_interactions, self.hessian_products):
                node_feats_hessian, sc = interaction(
                    node_attrs=data["node_attrs"],
                    node_feats=node_feats_hessian,
                    edge_attrs=edge_attrs,  # Use main graph edge attributes
                    edge_feats=edge_feats,  # Use main graph edge features
                    edge_index=data["edge_index"],  # Use main graph
                    cutoff=cutoff,  # Use main graph cutoff
                )
                node_feats_hessian = product(
                    node_feats=node_feats_hessian,
                    sc=sc,
                    node_attrs=data["node_attrs"],
                )
                # Append refined features to the list for Hessian prediction
                node_feats_list.append(node_feats_hessian)

        # Make l=0,1,2 node and edge features for the Hessian
        diag_feats_list = []
        off_diag_feats_list = []
        for i, node_feats in enumerate(node_feats_list):
            # Decide whether to grab features now or wait for the end
            is_last_layer = (i == len(self.interactions) - 1)
            
            if not self.hessian_use_last_layer_only or is_last_layer:
                # We need the spherical harmonics (edge_attrs) for the cross product
                # Ensure edge_attrs matches the lmax used in initialization
                # Diagonal Features (Per Node)
                # [BN, C] -> [BN, C']
                diag_feats = self.hessian_proj_nodes_layerwise(node_feats)
                
                # Off-Diagonal Features (Per Edge) - choose method
                if self.hessian_edge_feature_method == "message_passing":
                    # Use dedicated Hessian interaction block for message passing
                    
                    # Extract per-edge messages before aggregation using the Hessian-specific interaction
                    raw_messages = self._extract_raw_messages_from_interaction(
                        interaction=self.hessian_interaction,
                        node_attrs=data["node_attrs"],
                        node_feats=node_feats,
                        edge_attrs=edge_attrs_hessian,
                        edge_feats=edge_feats_hessian,
                        edge_index=edge_index_hessian,
                        cutoff=None,  # Cutoff already applied in radial embedding if needed
                    )
                    # [n_edges, irreps_mid]
                    
                    # Apply the interaction's linear layer to transform irreps_mid -> hidden_irreps
                    # raw_messages: [n_edges, irreps_mid] -> [n_edges, hidden_irreps]
                    raw_messages = self.hessian_interaction.linear(raw_messages)
                    
                    # Project raw messages to hessian output irreps
                    off_diag_feats = self.hessian_message_proj(raw_messages)
                    
                    # Apply edge-level gates for non-linearity (if enabled)
                    if self.hessian_use_edge_gates:
                        off_diag_feats = self.hessian_edge_gate(off_diag_feats)
                else:
                    # Use tensor product approach (edge_tp)
                    # j->i convention
                    j, i_idx = edge_index_hessian # [E], [E]  
                    h_j = node_feats[j] # [E, C]
                    
                    # Use both source and target node features if enabled
                    # When False, drop h_i entirely and use only h_j (symmetry enforced later)
                    if self.hessian_use_both_nodes:
                        h_i = node_feats[i_idx] # [E, C]
                        # Combine h_i and h_j: add and project
                        h_combined = self.hessian_node_combine(h_i + h_j)
                    else:
                        # Only use h_j, drop h_i entirely
                        h_combined = h_j
                    
                    # Compute tensor product with optional directional encoding
                    if self.hessian_use_directional_encoding:
                        # Include normalized edge direction r_ij explicitly
                        # First combine Y_ij and r_ij, then with h_j
                        combined_edge_attrs = self.edge_tp_directional(
                            edge_attrs_hessian,  # [E, L] - Y_ij
                            edge_vec_normalized_sh,  # [E, 3] - r_ij as spherical harmonics (1o)
                        )
                        # Then combine h_j with the enhanced edge features
                        off_diag_feats = self.edge_tp(
                            h_combined,
                            combined_edge_attrs,  # [E, L'] - enhanced edge features
                        )
                    else:
                        # Standard tensor product: TP(h_combined, Y_ij) -> produces 0e, 1e, 2e
                        off_diag_feats = self.edge_tp(
                            h_combined, 
                            edge_attrs_hessian, # [E, L]
                        )
                
                # Incorporate radial features if enabled (only for edge_tp method)
                if self.hessian_edge_feature_method == "edge_tp" and self.hessian_use_radial:
                    # Project radial features and use them to gate/modulate edge features
                    radial_proj = self.hessian_radial_proj(edge_feats_hessian)
                    # Element-wise multiplication (gating) with radial features
                    off_diag_feats = off_diag_feats * (1.0 + radial_proj)
                
                # Apply edge-level gates for non-linearity (if enabled)
                if self.hessian_use_edge_gates:
                    off_diag_feats = self.hessian_edge_gate(off_diag_feats)
                
                diag_feats_list.append(diag_feats)
                off_diag_feats_list.append(off_diag_feats)
        
        # Aggregation (if using multiple layers)
        diag_stacked = torch.stack(diag_feats_list, dim=0)  # [num_layers, BN, out_irreps]
        off_diag_stacked = torch.stack(off_diag_feats_list, dim=0)  # [num_layers, E, out_irreps]
        
        if self.hessian_aggregation == "learnable" and len(diag_feats_list) > 1:
            # Use learnable weights: softmax over learnable parameters
            weights = torch.softmax(self.hessian_layer_weights_param[:len(diag_feats_list)], dim=0)
            weights = weights.view(-1, 1, 1)  # [num_layers, 1, 1]
            # [BN, out_irreps]
            diag_out = (diag_stacked * weights).sum(dim=0)
            # [E, out_irreps]
            off_diag_out = (off_diag_stacked * weights).sum(dim=0)
        else:
            # Simple mean aggregation
            # [BN, out_irreps]
            diag_out = diag_stacked.mean(dim=0)
            # [E, out_irreps]
            off_diag_out = off_diag_stacked.mean(dim=0)
        
        # o3.Linear layer to project node_feats
        # onto the target irreps 1x0e + 1x1e + 1x2e = 9
        # Apply the linear projection to both diagonal and off-diagonal features
        # diag_feats: [BN, 9]
        diag_out = self.hessian_proj_nodes(diag_out)
        # off_diag_feats: [E, 9]
        off_diag_out = self.hessian_proj_edges(off_diag_out)
        
        # (E, 3, 3)
        l012_edge_feat_3x3 = irreps_to_cartesian_matrix(
            off_diag_out
        )  
        # (N, 3, 3)
        l012_node_feat_3x3 = irreps_to_cartesian_matrix(diag_out)

        hessian = blocks3x3_to_hessian(
            edge_index=edge_index_hessian,
            data=data,
            l012_edge_features=l012_edge_feat_3x3,
            l012_node_features=l012_node_feat_3x3,
        )

        return hessian

    def get_muon_param_groups(
        self,
        **kwargs,
    ):
        """
        Build parameter groups for MuonWithAuxAdam:
        - Muon group (use_muon=True): only parameters with ndim >= 2 inside hidden layers
          `blocks`.
        - Aux Adam group (use_muon=False): every other parameter in the model
          (embeddings, heads, biases/gains, blocks, etc.).

        Returns two param-group lists.
        """
        muon_params = []
        adam_params = []

        for name, param in self.named_parameters():
            if name.startswith("interactions.") and param.ndim >= 2:
                muon_params.append(param)
            else:
                adam_params.append(param)

        return muon_params, adam_params

@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale, shift=atomic_inter_shift
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        # added for HIP Hessian prediction
        predict_hessian: bool = False, 
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        self.atomic_numbers: [Z_max]
        node_attrs: [B*N, Z_max]
        edge_index: [2, E]
        """
        # Setup
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange.to(torch.int64)
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads.to(torch.int64)
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, num_heads]

        # Embeddings
        # [3, D1]
        node_feats = self.node_embedding(data["node_attrs"])
        # [E, D2]
        edge_attrs = self.spherical_harmonics(vectors)
        # [E, D3], None 
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # Embeddings of additional features
        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            # [N, D, L]
            # where L=1+3+5+7+...
            # sc = TensorProduct on skip connection
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            # [3, D1]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)
        
        # all ?x0e
        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange, node_heads
                ]
            )
        
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        # compute forces via autograd
        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=inter_e,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces or compute_atomic_stresses,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        
        if predict_hessian:
            hessian = self._predict_hessian_hip(
                data, node_feats_list, edge_attrs, edge_feats, cutoff
            )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }
        


@compile_mode("script")
class AtomicDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[
            None
        ],  # Just here to make it compatible with energy models, MUST be None
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        assert atomic_energies is None

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[1]
                )  # Select only l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=True
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
        # added for HIP Hessian prediction
        predict_hessian: bool = False, # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the dipoles
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        output = {
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output


@compile_mode("script")
class AtomicDielectricMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[
            None
        ],  # Just here to make it compatible with energy models, MUST be None
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_type: Optional[str] = "bessel",
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
        dipole_only: Optional[bool] = True,  # pylint: disable=unused-argument
        use_polarizability: Optional[bool] = True,  # pylint: disable=unused-argument
        means_stds: Optional[Dict[str, torch.Tensor]] = None,  # pylint: disable=W0613
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )

        # Predefine buffers to be TorchScript-safe
        self.register_buffer("dipole_mean", torch.zeros(3))
        self.register_buffer("dipole_std", torch.ones(3))
        self.register_buffer(
            "polarizability_mean", torch.zeros(3, 3)
        )  # 3x3 matrix flattened
        self.use_polarizability = use_polarizability
        self.register_buffer("polarizability_std", torch.ones(3, 3))
        self.register_buffer("change_of_basis", get_change_of_basis())
        # self.register_buffer("mean_polarizability_sh", torch.zeros(6))
        # self.register_buffer("std_polarizability_sh", torch.ones(6))
        if means_stds is not None:
            if "dipole_mean" in means_stds:
                self.dipole_mean.data.copy_(means_stds["dipole_mean"])
            if "dipole_std" in means_stds:
                self.dipole_std.data.copy_(means_stds["dipole_std"])
            if "polarizability_mean" in means_stds:
                self.polarizability_mean.data.copy_(means_stds["polarizability_mean"])
            if "polarizability_std" in means_stds:
                self.polarizability_std.data.copy_(means_stds["polarizability_std"])
            # if "mean_polarizability_sh" in means_stds:
            #    self.mean_polarizability_sh.data.copy_(means_stds["mean_polarizability_sh"])
            # if "std_polarizability_sh" in means_stds:
            #    self.std_polarizability_sh.data.copy_(means_stds["std_polarizability_sh"])'''
        assert atomic_energies is None
        # self.use_polarizability = use_polarizability
        # self.use_dipole = use_dipole

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearDipolePolarReadoutBlock(hidden_irreps, use_polarizability=True)
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                # does it always do polar and dipole together?
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                # hidden_irreps_out = str(
                #     hidden_irreps[1]
                # )  # Select only l=1 vectors for last layer
                hidden_irreps_out = (
                    hidden_irreps  # this is different in the AtomicDipoleMACE
                )
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipolePolarReadoutBlock(
                        hidden_irreps_out,
                        MLP_irreps,
                        gate,
                        use_polarizability=True,
                    )
                )
                # print("Nonlinear irrpes: ", hidden_irreps_out, MLP_irreps)
                # exit()
            else:
                self.readouts.append(
                    LinearDipolePolarReadoutBlock(
                        hidden_irreps,
                        # use_charge=True,
                        use_polarizability=True,
                    )
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,  # pylint: disable=W0613
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_dielectric_derivatives: bool = False,  # no training on derivatives
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
        # added for HIP Hessian prediction
        predict_hessian: bool = False, # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        assert compute_displacement is False
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        charges = []
        dipoles = []
        polarizabilities = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )

            node_out = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            charges.append(node_out[:, 0])

            if self.use_polarizability:
                node_dipoles = node_out[:, 2:5]
                node_polarizability = torch.cat(
                    (node_out[:, 1].unsqueeze(-1), node_out[:, 5:]), dim=-1
                )
                polarizabilities.append(node_polarizability)
                dipoles.append(node_dipoles)
            else:
                raise ValueError(
                    "Polarizability is not used in this model, but it is required for the AtomicDielectricMACE."
                )
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        atomic_charges = torch.stack(charges, dim=-1).sum(-1)  # [n_nodes,]
        # The idea is to normalize the charges so that they sum to the net charge in the system before predicting the dipole.
        total_charge_excess = scatter_mean(
            src=atomic_charges, index=data["batch"], dim_size=num_graphs
        ) - (data["total_charge"] / num_atoms)
        atomic_charges = atomic_charges - total_charge_excess[data["batch"]]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole_polar(
            charges=atomic_charges,  # or data["charges"], ?????
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        if self.use_polarizability:
            # Compute the polarizabilities
            contributions_polarizabilities = torch.stack(
                polarizabilities, dim=-1
            )  # [n_nodes,6,n_contributions]
            atomic_polarizabilities = torch.sum(
                contributions_polarizabilities, dim=-1
            )  # [n_nodes,6]
            total_polarizability_spherical = scatter_sum(
                src=atomic_polarizabilities,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,6]
            total_polarizability = spherical_to_cartesian(
                total_polarizability_spherical, self.change_of_basis
            )

            if compute_dielectric_derivatives:
                dmu_dr = compute_dielectric_gradients(
                    dielectric=total_dipole,
                    positions=data["positions"],
                )
                dalpha_dr = compute_dielectric_gradients(
                    dielectric=total_polarizability.flatten(-2),
                    positions=data["positions"],
                )
            else:
                dmu_dr = None
                dalpha_dr = None
        else:
            if compute_dielectric_derivatives:
                dmu_dr = compute_dielectric_gradients(
                    dielectric=total_dipole,
                    positions=data["positions"],
                )
            else:
                dmu_dr = None
            total_polarizability = None
            total_polarizability_spherical = None
            dalpha_dr = None

        output = {
            "charges": atomic_charges,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
            "polarizability": total_polarizability,
            "polarizability_sh": total_polarizability_spherical,
            "dmu_dr": dmu_dr,
            "dalpha_dr": dalpha_dr,
        }
        return output


@compile_mode("script")
class EnergyDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out, MLP_irreps, gate, dipole_only=False
                    )
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False)
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
        # added for HIP Hessian prediction
        predict_hessian: bool = False, # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_atoms_arange = torch.arange(data["positions"].shape[0])
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, data["head"][data["batch"]]
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_out = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            # node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies = node_out[:, 0]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        contributions_dipoles = torch.stack(
            dipoles, dim=-1
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data["batch"].unsqueeze(-1),
            dim=0,
            dim_size=num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data["charges"],
            positions=data["positions"],
            batch=data["batch"],
            num_graphs=num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, virials, stress, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }
        return output
