"""Custom layers, blocks, and network components for working on graphs. These modules
are designed to be combined and chained together.

- Each module contains the attributes inpt_dim and outp_dim which allows them to be chained
"""

import math
from copy import deepcopy
from typing import Union

import torch as T
import torch.nn as nn

from ..distances import knn, masked_dist_matrix
from ..gnets.gn_blocks import GNBlock
from ..gnets.graphs import GraphBatch
from ..modules import DenseNetwork
from ..torch_utils import pass_with_mask, smart_cat
from ..utils import merge_dict


class EmbeddingLayer(nn.Module):
    """A layer for embedding the nodes into different space using an dense network.

    - Does not change the edges, globals, adjmat or mask
    - Each node operation is independant, no interaction
    - The embedding is conditioned on the graph globs and cndts
    """

    def __init__(
        self, inpt_dim: list, ctxt_dim: int = 0, net_kwargs: dict | None = None
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g]
        kwargs:
            ctxt_dim: The dimension of the context tensor
            net_kwargs: A dictionary for the embedding mlp
        """
        super().__init__()

        # Save dict kwargs
        net_kwargs = net_kwargs or {}

        # Store class attributes
        self.inpt_dim = inpt_dim
        self.ctxt_dim = ctxt_dim

        # The node embedding mlp (default output is the same as input)
        self.dense_net = DenseNetwork(
            inpt_dim=inpt_dim[1], ctxt_dim=inpt_dim[2] + ctxt_dim, **net_kwargs
        )

        # The output dimension comes from input with new node features
        self.outp_dim = [inpt_dim[0], self.dense_net.outp_dim, inpt_dim[2]]

    def forward(self, graph: GraphBatch, ctxt: T.Tensor | None = None) -> GraphBatch:
        """Forward pass for EmbeddingLayer."""
        graph.nodes = pass_with_mask(
            graph.nodes,
            self.dense_net,
            mask=graph.mask,
            high_level=smart_cat([graph.globs, ctxt]),
        )
        return graph

    def __repr__(self):
        return f"NodeEmbed[{self.dense_net.one_line_string()}]"


class EdgeBuilder(nn.Module):
    """Replaces a graph's adjacency matrix and the sometimes the edge feature matrix
    using a variety of different methods.

    The adjacency matrix is created using the distances between nodes
    - A subset of the node features are used as coordinates to calculate distances
    - The features used as co-ords can be stripped from the nodes

    The edge feature matrix changes with adjacency, it can be constructed using
    - The original edges with 0's for the brand new connections
    - Edge features may be completely replaced using distance matrix
    """

    def __init__(
        self,
        inpt_dim: list,
        clus_type: str = "knn",
        clus_val: int = 4,
        edge_type: str = "same",
        crd_frc: float = 0.5,
        strip: bool = False,
        measure: str = "eucl",
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g,c]
        kwargs:
            clus_type: A string to indicate the type of edge building to perform
                -> knn: A node can only send information to its k nearest neighbours
                -> thresh: Connects two edges if their 2-norm distance is below a thresh
                -> fc: Creates a fully connected graph with no self connections
            clus_val: A value to apply to the clustering method
                -> K if clus_type is 'knn'
                -> threshold if clus_type is 'thresh'
            edge_type: How the new edges are created
                -> same: Keep original edge features, 0 pad in the new ones
                -> diff: Replace all edge features with the difference matrix
                -> dist: Replace all edge features with the distance matrix
            crd_frc: The fraction of node features to use in clustering (rounds up)
            strip: Remove all features used to create the matrix from the nodes
            measure: How the distance between nodes is measured
                -> eucl: The euclidean distance between node features
                -> dot: The cosine similarity distance between features
        """
        super().__init__()

        # Configuration checks
        assert clus_type in ["knn", "thresh", "fc"]
        assert edge_type in ["same", "dist", "diff"]
        if clus_type == "fc" and strip:
            raise ValueError(
                "Can not use the strip option if building a fully connected adjmat"
            )

        # We store the input, output dimensions to query them later
        self.inpt_dim = inpt_dim
        self.outp_dim = inpt_dim.copy()  # Assuming dims dont change for now

        # Save the class attributes
        self.clus_type = clus_type
        self.clus_val = clus_val
        self.edge_type = edge_type
        self.crd_frc = crd_frc
        self.strip = strip
        self.measure = measure

        # Change the coord fraction to an interger number of features
        if crd_frc <= 1:
            self.n_crd = math.ceil(crd_frc * inpt_dim[1])
        else:
            self.n_crd = crd_frc

        # Change the output dimension based on how we construct edge and node features
        if edge_type == "dist":
            self.outp_dim[0] = 1
        elif edge_type == "diff":
            self.outp_dim[0] = self.n_crd
        if strip:
            self.outp_dim[1] = self.outp_dim[1] - self.n_crd

    def forward(self, graph: GraphBatch, ctxt=None) -> GraphBatch:
        """Forward pass for EdgeBuilder."""

        # Get the subset of node features to use as coordinates
        coords = graph.nodes[..., : self.n_crd]

        # Create the fully connected adjacency and distance matrices
        distmat, new_adjmat = masked_dist_matrix(
            coords,
            graph.mask,
            track_grad=(self.edge_type == "dist"),
            measure=self.measure,
            pad_val=T.inf if self.measure == "eucl" else -T.inf,
        )

        # Trim the adjmat using the cluster type
        if self.clus_type == "knn":
            new_adjmat = knn(distmat, self.clus_val, top_k=(self.measure == "dot"))
        elif self.clus_type == "thresh":
            if self.measure == "eucl":
                new_adjmat = distmat < self.clus_val
            else:
                new_adjmat = distmat > self.clus_val

        # Create the new edge features according to the new matrix
        if self.edge_type == "dist":
            new_edges = distmat[new_adjmat].unsqueeze(-1)

        # Use use the existing edges, replacing the new ones with padded values
        elif self.edge_type == "same":
            # Create the new padded edges
            new_edges = T.zeros(
                (new_adjmat.sum(), graph.edges.shape[-1]),
                dtype=T.float,
                device=graph.edges.device,
            )

            # If previous edges existed then we need to keep those that survived
            if graph.edges.nelement():
                pad = graph.adjmat[new_adjmat]
                srvd = new_adjmat[graph.adjmat]
                new_edges[pad] = graph.edges[srvd]

        # Update all the graph features
        graph.nodes = graph.nodes[:, :, self.n_crd :] if self.strip else graph.nodes
        graph.adjmat = new_adjmat
        graph.edges = new_edges

        # Return the new graph object
        return graph

    def __repr__(self):
        string = f"EdgeBuilder({self.clus_type}-{self.clus_val}, n_crd={self.n_crd}"
        string += f", measure = {self.measure}"
        if self.strip:
            string += ", strip"
        if self.edge_type != "same":
            string += f", new_edges={self.edge_type}"
        string += ")"
        return string


class GraphNeuralNetwork(nn.Module):
    """A stack of GNBlocks with optional Edge-Building layers in between.

    Graph -> Graph.
    """

    def __init__(
        self,
        inpt_dim: list,
        ctxt_dim: int = 0,
        num_blocks: int = 1,
        ebl_every: int = 0,
        start_with_ebl: bool = False,
        ebl_kwargs: Union[dict, list] | None = None,
        gnb_kwargs: Union[dict, list] | None = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g]
        kwargs:
            ctxt_dim: The size of the contect tensor
            depth: The number of GNBlocks to use
            ebl_every: How often an edge building layer is inserted inbetween GNBlocks
            start_with_ebl: If the stack begins with an edge building layer
            gnb_kwargs: Arguments for the graph network blocks
            ebl_kwargs: Arguments for the edge building layers
        """
        super().__init__()

        # Dict default kwargs
        ebl_kwargs = ebl_kwargs or {}
        gnb_kwargs = gnb_kwargs or {}

        # Generate lists of dictionaries using first as template
        if isinstance(ebl_kwargs, list):
            if len(ebl_kwargs) != num_blocks - 1 + start_with_ebl:
                raise ValueError("Length of EdgeBuilder kwargs does not match depth")
            ebl_list = ebl_kwargs.copy()
            for i in range(1, num_blocks):
                ebl_list[i] = merge_dict(ebl_list[0], ebl_list[i])
        else:
            ebl_list = num_blocks * [ebl_kwargs]

        if isinstance(gnb_kwargs, list):
            if len(gnb_kwargs) != num_blocks:
                raise ValueError("Length of GNBlock kwargs does not match depth")
            gnb_list = gnb_kwargs.copy()
            for i in range(1, num_blocks):
                gnb_list[i] = merge_dict(gnb_list[0], gnb_list[i])
        else:
            gnb_list = num_blocks * [gnb_kwargs]

        # Store the input dimensions of the entire stack
        self.inpt_dim = inpt_dim

        # Initialise all the blocks as a module list
        self.blocks = nn.ModuleList()

        # The first (and sometimes only) edge building step
        if start_with_ebl:
            self.blocks.append(EdgeBuilder(inpt_dim, **ebl_list[0]))
            cur_dim = self.blocks[-1].outp_dim
        else:
            cur_dim = self.inpt_dim

        # The first (and sometimes only) GN block
        self.blocks.append(GNBlock(cur_dim, ctxt_dim=ctxt_dim, **gnb_list[0]))

        # The extra blocks
        for layer in range(1, num_blocks):
            # Add an edge building layer
            if ebl_every > 0 and layer % ebl_every == 0:
                self.blocks.append(
                    EdgeBuilder(self.blocks[-1].outp_dim, **ebl_list[layer])
                )

            # Add a GN block
            self.blocks.append(
                GNBlock(self.blocks[-1].outp_dim, ctxt_dim=ctxt_dim, **gnb_list[layer])
            )

        # Calculate the output dimension using the final layer
        self.outp_dim = self.blocks[-1].outp_dim

    def forward(self, graph: GraphBatch, ctxt=None) -> GraphBatch:
        """Forward pass for GraphNetwork."""

        # Pass through each of the submodules
        for module in self.blocks:
            graph = module(graph, ctxt=ctxt)

        return graph


class GraphVectorGenerator(nn.Module):
    """A simple graph generator which takes in a singe vector. vector to graph.

    Creates the point cloud using
       - no edge features
       - gaussian noise for nodes
       - no globals
       - pre-determined mask (passed in forward call)
       - fully connected adjmat (can be overwritten by GNN layers)
    Applies GN updates to the graph using the vector and any other context as ctxt
    """

    def __init__(
        self,
        inpt_dim: int,
        ctxt_dim: int = 0,
        node_init_dim: int = 4,
        gnn_kwargs: dict | None = None,
    ) -> None:
        """
        args:
            inpt_dim: The size of the input vector.
        kwargs:
            ctxt_dim: The dimension of the contextual tensor for conditional generation.
            node_init_dim: The initial dimension of the nodes (if 0 takes from output).
            dns_kwargs: The keyword for a the initial dense network for vector.
            gnn_kwargs: The kwargs dict for the GNN layers.
            emb_kwargs: The kwargs dict for the output embedding mlp.
        """
        super().__init__()

        # Dict default kwargs, copy makes it safe when we change output
        gnn_kwargs = gnn_kwargs.copy() or {}

        # Save the class attributes
        self.inpt_dim = inpt_dim
        self.node_init_dim = node_init_dim
        self.ctxt_dim = ctxt_dim

        # Create the graph network, takes initial vec and ctxt as conditioning
        self.gnn = GraphNeuralNetwork(
            inpt_dim=[0, self.node_init_dim, 0],
            ctxt_dim=inpt_dim + ctxt_dim,
            **gnn_kwargs,
        )

        # Set the final output dimension of this module
        self.outp_dim = self.gnn.outp_dim

    def _create_init_graph(
        self, inputs: T.Tensor, target_mask: T.BoolTensor, ctxt: T.Tensor
    ) -> GraphBatch:
        """Return the initial random graph."""

        # Get the sizes and kwargs for the graph batch
        b_size = len(inputs)
        n_nodes = target_mask.shape[-1]
        edge_size = (n_nodes * n_nodes, 0)
        node_size = (b_size, n_nodes, self.node_init_dim)
        glob_size = (b_size, 0)
        adjm_size = (b_size, n_nodes, n_nodes)
        kwargs = {"dtype": T.float32, "device": inputs.device}

        # Initialise the random graph batch
        inputs = GraphBatch(
            edges=T.zeros(edge_size, **kwargs),
            nodes=T.randn(node_size, **kwargs),
            globs=T.zeros(glob_size, **kwargs),
            adjmat=T.ones(adjm_size, dtype=T.bool, device=inputs.device),
            mask=target_mask,
            dev=inputs.device,
        )

        # All nodes were just initialised as random! Apply the masking!
        inputs.nodes = inputs.nodes * target_mask.unsqueeze(-1)

        return inputs

    def forward(
        self, inputs: T.Tensor, target_mask: T.BoolTensor, ctxt: T.Tensor | None = None
    ) -> GraphBatch:
        """Given some tensors, create the output graph object
        args:
            inputs: The input batch of tensors
            target_mask: The requested mask of the generated graph batch
        kwargs:
            ctxt: The input context tensors
        """
        # Create the random graph with correct masking and inputs as globals
        graph = self._create_init_graph(inputs, target_mask, ctxt)

        # Pass the random graph through the graph network to update
        graph = self.gnn(graph, ctxt=smart_cat([inputs, ctxt]))

        return graph


class FullGraphVectorGenerator(nn.Module):
    """A GVG with added input and output embedding networks (for the nodes) Vector to
    Graph.

    1)  Embeds the input vector into a higher dimensional space based on model_dim using
    a dense network. 2)  Passes this through a GVG to get a graph output 3) Passes the
    sequence through an embedding dense network with vector as context
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        gvg_kwargs: dict | None = None,
        vect_embd_kwargs: dict | None = None,
        outp_embd_kwargs: dict | None = None,
    ) -> None:
        """
        args:
            inpt_dim: The size of the input vector.
            outp_dim: The dimension of the output nodes of the PC
        kwargs:
            ctxt_dim: Dim. of the context vector to pass to the embedding nets
            gvg_kwargs: Keyword arguments to pass to the gvg constructor
            vec_embd_kwargs: Keyword arguments for vector ff embedder
            out_embd_kwargs: Keyword arguments for output node ff embedder
        """
        super().__init__()

        # Safe default dict arguments
        gvg_kwargs = gvg_kwargs or {}
        vect_embd_kwargs = vect_embd_kwargs or {}
        outp_embd_kwargs = outp_embd_kwargs or {}

        # Save the class attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # The initial dense network
        self.vec_embd = DenseNetwork(
            inpt_dim=inpt_dim, ctxt_dim=ctxt_dim, **vect_embd_kwargs
        )

        # The graph generator
        self.gvg = GraphVectorGenerator(
            inpt_dim=self.vec_embd.outp_dim, ctxt_dim=ctxt_dim, **gvg_kwargs
        )

        # The output embedding network
        self.outp_embd = EmbeddingLayer(
            inpt_dim=self.gvg.outp_dim, ctxt_dim=ctxt_dim, **outp_embd_kwargs
        )

    def forward(
        self, vec: T.Tensor, mask: T.BoolTensor, ctxt: T.Tensor | None = None
    ) -> tuple:
        """Pass the input through all layers sequentially."""
        vec = self.vec_embd(vec, ctxt=ctxt)
        graph = self.gvg(vec, mask, ctxt=ctxt)
        graph = self.outp_embd(graph, ctxt=ctxt)
        return graph


class FullGraphVectorEncoder(nn.Module):
    """A graph network encoder which produces a vector representation.

    Graph -> Vector.
    """

    def __init__(
        self,
        inpt_dim: list,
        outp_dim: int,
        ctxt_dim: int = 0,
        gnn_kwargs: dict | None = None,
        dns_kwargs: dict | None = None,
    ) -> None:
        """
        args:
            inpt_dim: The dimensions of the input graph [e,n,g]
            outp_dim: The size of the encoded output
        kwargs:
            ctxt_dim: Dim. of the context vector to pass to all nets
            gnn_kwargs: The keyword arguments for the GraphNeuralNetwork
            dns_kwargs: The keyword arguments to for the output dense network
        """
        super().__init__()

        # Dict default kwargs, copy makes it safe when we change output
        gnn_kwargs = deepcopy(gnn_kwargs) or {}
        dns_kwargs = deepcopy(dns_kwargs) or {}

        # Save the class attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        # The series of modules that make up the network
        self.gnn = GraphNeuralNetwork(inpt_dim, ctxt_dim, **gnn_kwargs)
        self.dns = DenseNetwork(
            inpt_dim=self.gnn.outp_dim[2],
            ctxt_dim=ctxt_dim,
            outp_dim=outp_dim,
            **dns_kwargs,
        )

    def forward(
        self,
        inputs: GraphBatch,
        ctxt: T.Tensor | None = None,
        return_nodes: bool = False,
    ) -> T.Tensor:
        """Encode a graph batch."""
        inputs = self.gnn(inputs, ctxt=ctxt)
        vec = self.dns(inputs.globs, ctxt=ctxt)
        if return_nodes:
            return vec, inputs.nodes
        return vec
