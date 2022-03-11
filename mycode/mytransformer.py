from fairseq.models.transformer import (
    TransformerModel, 
    base_architecture, 
    TransformerDecoder
)
from fairseq.models import (
    register_model, 
    register_model_architecture
)

from fairseq.modules import TransformerDecoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

import torch
from fairseq import utils
import torch.nn as nn
import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
import math

@register_model('Conf_transformer')
class Conf_TransformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return Conf_TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # print("Conf_transformer")
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError


class Conf_TransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.conf_layer=nn.Linear(self.output_embed_dim, 1, bias=True)
        self.activation_layer=nn.Sigmoid()

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            c = self.activation_layer(self.conf_layer(torch.mean(torch.stack(extra["inner_states"][1:4]),dim=0).transpose(0, 1))) # [batch, tgt_len, 1]
            x = self.output_layer(x) # [batch, tgt_len, vocab]
            pad = torch.zeros(extra["attn"][0].shape[0], extra["attn"][0].shape[1], extra["attn"][0].shape[2]-1).cuda()
            extra["attn"]=[torch.cat([pad, c], -1)]
            del pad
        return x, extra, c

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]], Tensor],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        # print('ConNet Decoder')
        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        # if log_probs:
        #     return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        # else:
        #     return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
      
        # logits = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)*torch.pow(2,net_output[-1])
        logits = utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        # logits = torch.where(net_output[-1].repeat(1,1,logits.shape[-1])>0.5,logits,1e-11*torch.ones_like(logits).cuda())
        if log_probs:
            logits = torch.log(logits)
        
        return logits

@register_model_architecture('Conf_transformer', 'my_arch')
def my_hyperparameters(args):
    base_architecture(args)

