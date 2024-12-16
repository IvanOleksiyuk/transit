from torch import nn
from mltools.mlp import MLP
import torch 


class MLP(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks.

    Supports context injection layers.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: int | list = 32,
        num_blocks: int = 1,
        n_lyr_pbk: int = 1,
        act_h: str = "lrlu",
        act_o: str = "none",
        do_out: bool = True,
        nrm: str = "none",
        drp: float = 0,
        drp_on_output: bool = False,
        nrm_on_output: bool = False,
        do_res: bool = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        ctxt_in_out: bool = False,
        do_bayesian: bool = False,
        init_zeros: bool = False,
        use_bias: bool = True,
    ) -> None:
        """Initialise the MLP.

        Parameters
        ----------
        inpt_dim : int
            The number of input features
        outp_dim : int, optional
            The number of output features, by default 0
        ctxt_dim : int, optional
            The number of contextual features, by default 0
        hddn_dim : int | list, optional
            The number of hidden features in each block, by default 32
        num_blocks : int, optional
            The number of hidden blocks, by default 1.
            Ignored if hddn_dim is a list.
        n_lyr_pbk: int, optional
            The number of layers in each hidden block, by default 1
        act_h : str, optional
            The activation function for the hidden blocks, by default "lrlu"
        act_o : str, optional
            The activation function for the output block, by default "none"
        do_out : bool, optional
            If to include an output block, by default True
        nrm : str, optional
            The normalisation for the hidden blocks, by default "none"
        drp : float, optional
            The dropout probability for the hidden blocks, by default 0
        drp_on_output : bool, optional
            If to apply dropout to the output block, by default False
        nrm_on_output : bool, optional
            If to apply normalisation to the output block, by default False
        do_res  : bool, optional
            If to include residual connections, by default False
        ctxt_in_inpt    : bool, optional
            If to concatenate the context to the input layer, by default True
        ctxt_in_hddn    : bool, optional
            If to concatenate the context to the hidden layers, by default False
        ctxt_in_out    : bool, optional
            If to concatenate the context to the output layer, by default False
        do_bayesian : bool, optional
            If to fill the block with bayesian linear layers, by default False
        init_zeros  : bool, optional
            If the final layer parameters in each MLP block are set to zero
            Does not apply to bayesian layers
            Will also prevent normalisation
        use_bias: bool, optional
            If the linear layers use bias terms
        """
        super().__init__()

        # Check that the context is used somewhere
        if ctxt_dim:
            if not ctxt_in_inpt and not ctxt_in_hddn and not ctxt_in_out:
                raise ValueError("Network has context inputs but nowhere to use them!")

        # We store the input, hddn (list), output, and ctxt dims to query them later
        self.inpt_dim = inpt_dim
        if not isinstance(hddn_dim, int):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]
        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out

        # Necc for this module to work with the nflows package
        self.hidden_features = self.hddn_dim[-1]

        # Input MLP block
        self.input_block = MLPBlock(
            inpt_dim=self.inpt_dim,
            outp_dim=self.hddn_dim[0],
            ctxt_dim=self.ctxt_dim if ctxt_in_inpt else 0,
            act=act_h,
            nrm=nrm,
            drp=drp,
            do_bayesian=do_bayesian,
            use_bias=use_bias,
        )

        # All hidden blocks as a single module list
        self.hidden_blocks = []
        if self.num_blocks > 1:
            self.hidden_blocks = nn.ModuleList()
            for h_1, h_2 in zip(self.hddn_dim[:-1], self.hddn_dim[1:]):
                self.hidden_blocks.append(
                    MLPBlock(
                        inpt_dim=h_1,
                        outp_dim=h_2,
                        ctxt_dim=self.ctxt_dim if ctxt_in_hddn else 0,
                        n_layers=n_lyr_pbk,
                        act=act_h,
                        nrm=nrm if nrm_inside else "none",
                        drp=drp,
                        do_res=do_res,
                        init_zeros=init_zeros,
                        do_bayesian=do_bayesian,
                        use_bias=use_bias,
                    )
                )

        # Output block
        if do_out:
            self.output_block = MLPBlock(
                inpt_dim=self.hddn_dim[-1],
                outp_dim=self.outp_dim,
                ctxt_dim=self.ctxt_dim if ctxt_in_out else 0,
                act=act_o,
                do_bayesian=do_bayesian,
                init_zeros=init_zeros,
                nrm=nrm if nrm_on_output else "none",
                drp=drp if drp_on_output else 0,
                use_bias=use_bias,
            )

    def forward(
        self,
        inputs: T.Tensor,
        ctxt: T.Tensor | None = None,
        context: T.Tensor | None = None,
    ) -> T.Tensor:
        """Pass through all layers of the dense network."""

        # Use context as a synonym for ctxt (normflow compatibility)
        if context is not None:
            ctxt = context

        # Reshape the context if it is available. Equivalent to performing
        # multiple ctxt.unsqueeze(1) until the dim matches the input.
        # Batch dimension is kept the same.
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)

        # Pass through the input block
        inputs = self.input_block(inputs, ctxt)

        # Pass through each hidden block
        for h_block in self.hidden_blocks:  # Context tensor will only be used if
            inputs = h_block(inputs, ctxt)  # block was initialised with a ctxt dim

        # Pass through the output block
        if self.do_out:
            inputs = self.output_block(inputs)

        return inputs

    def __repr__(self):
        string = ""
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, h_block in enumerate(self.hidden_blocks):
            string += f"  (h-{i+1}): " + repr(h_block) + "\n"
        if self.do_out:
            string += "  (out): " + repr(self.output_block)
        return string

    def one_line_string(self):
        """Return a one line string that sums up the network structure."""
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += ">"
        string += str(self.input_block.outp_dim) + ">"
        if self.num_blocks > 1:
            string += ">".join(
                [
                    str(layer.out_features)
                    for hidden in self.hidden_blocks
                    for layer in hidden.block
                    if isinstance(layer, nn.Linear)
                ]
            )
            string += ">"
        if self.do_out:
            string += str(self.outp_dim)
        return string