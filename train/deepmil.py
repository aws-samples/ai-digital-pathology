"""
Copyright (c) Owkin Inc.
This source code is licensed under the GNU GENERAL PUBLIC LICENSE v3 license found in the
LICENSE file in the root directory of this source tree.

Implementation of DeepMIL model, as proposed by
Ilse et al. 2018, Attention-based Deep Multiple Instance Learning.
"""
from typing import Optional, List, Tuple, Union
import torch

class MLP(torch.nn.Sequential):
    """
    MLP Module

    Parameters
    ----------
    in_features: int
    out_features: int
    hidden: Optional[List[int]] = None
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)


class TilesMLP(torch.nn.Module):
    """
    MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    hidden: Optional[List[int]] = None:
        Number of hidden layers and their respective number of features.
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    activation: torch.nn.Module = torch.nn.Sigmoid()
        MLP activation function
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
    ):
        super(TilesMLP, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        if hidden is not None:
            for h in hidden:
                self.hidden_layers.append(
                    MaskedLinear(in_features, h, bias=bias, mask_value="-inf")
                )
                self.hidden_layers.append(activation)
                in_features = h

        self.hidden_layers.append(torch.nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.
        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x
    
class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.
    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None):
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, mask_value={}, bias={}".format(
            self.in_features, self.out_features, self.mask_value, self.bias is not None
        )
    
class GatedAttention(torch.nn.Module):
    """
    Gated Attention, as defined in https://arxiv.org/abs/1802.04712.
    Permutation invariant Layer on dim 1.

    Parameters
    ----------
    d_model: int = 128
    """

    def __init__(
        self,
        d_model: int = 128,
    ):
        super(GatedAttention, self).__init__()

        self.att = torch.nn.Linear(d_model, d_model)
        self.gate = torch.nn.Linear(d_model, d_model)
        self.w = MaskedLinear(d_model, 1, "-inf")

    def attention(
        self,
        v: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Gets attention logits.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """

        h_v = self.att(v)
        h_v = torch.tanh(h_v)

        u_v = self.gate(v)
        u_v = torch.sigmoid(u_v)

        attention_logits = self.w(h_v * u_v, mask=mask)
        return attention_logits

    def forward(
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights


class DeepMIL(torch.nn.Module):
    """
    Deep MIL classification model.
    https://arxiv.org/abs/1802.04712

    Parameters
    ----------
    in_features: int
    out_features: int = 1
    d_model_attention: int = 128
    tiles_mlp_hidden: Optional[List[int]] = None
    mlp_hidden: Optional[List[int]] = None
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
    bias: bool = True
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_model_attention: int = 128,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        super(DeepMIL, self).__init__()

        self.tiles_emb = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=d_model_attention,
        )

        self.attention_layer = GatedAttention(d_model=d_model_attention)

        mlp_in_features = d_model_attention

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            activation=mlp_activation,
        )

    def score_model(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Gets attention logits.
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        tiles_emb = self.tiles_emb(x, mask)
        attention_logits = self.attention_layer.attention(tiles_emb, mask)
        return attention_logits

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        logits, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, OUT_FEATURES), (B, N_TILES)
        """
        tiles_emb = self.tiles_emb(x, mask)
        scaled_tiles_emb, _ = self.attention_layer(tiles_emb, mask)

        logits = self.mlp(scaled_tiles_emb)

        return logits