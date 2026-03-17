import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import TopKSaeConfig


class TopKSaeLayer(BaseTunerLayer):
    """
    Sparse Autoencoder Layer for PEFT.
    """

    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("sae_encoder", "sae_W_dec", "sae_b_dec", 
                                            )    # "sae_patch_map", "sae_token_map"
    # All names of other parameters that may contain adapter-related parameters
    # other_param_names: tuple[str, ...] = ("k",)       # no need to move int k to device, and to allow shift k training

    def __init__(self, base_layer: nn.Module, *args, **kwargs):
        self.base_layer = base_layer
        self.sae_encoder = nn.ModuleDict({})
        self.sae_W_dec = nn.ParameterDict({})
        self.sae_b_dec = nn.ParameterDict({})
        # self.sae_patch_map = nn.ParameterDict({})
        # self.sae_token_map = nn.ParameterDict({})
        self.k = {}
        self.num_latents = 0
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Module):
            in_features, out_features = (
                base_layer.mlp.down_proj.in_features,
                base_layer.mlp.down_proj.out_features,
            )
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(
                base_layer, "out_features"
            ):
                in_features, out_features = (
                    base_layer.in_features,
                    base_layer.out_features,
                )
            else:
                warnings.warn(
                    f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.",
                    UserWarning,
                )
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name: str,
        k: int,
        num_latents: int,
        expansion_factor: int,
        **kwargs,
    ):
        if num_latents == 0:
            num_latents = expansion_factor * self.out_features
        self.num_latents = num_latents

        self.sae_encoder[adapter_name] = nn.Linear(
            self.out_features, num_latents, bias=True
        )
        # When we init new adapters, we copy the weights from the encoder for decoder
        self.sae_W_dec[adapter_name] = nn.Parameter(
            self.sae_encoder[adapter_name].weight.data.clone()
        )
        self.sae_b_dec[adapter_name] = nn.Parameter(
            torch.zeros(self.out_features, dtype=torch.float32)
        )
        self.k[adapter_name] = k
        
        # print("debug: update_layer")
        # print(f"debug: self.out_features={self.out_features} self.num_latents={self.num_latents} k={k}")
        # print(f"debug: sae_encoder dtype={self.sae_encoder[adapter_name].weight.dtype}, shape={self.sae_encoder[adapter_name].weight.shape}")
        # print(f"debug: W_dec dtype={self.sae_W_dec[adapter_name].dtype}, shape={self.sae_W_dec[adapter_name].shape}")
        # print(f"debug: bias dtype={self.sae_b_dec[adapter_name].dtype}, shape={self.sae_b_dec[adapter_name].shape}")

        # add cross-attention: out_features -> out_features
        # self.sae_patch_map[adapter_name] = nn.Linear(
        #     self.out_features, self.out_features, bias=False
        # )
        # self.sae_token_map[adapter_name] = nn.Linear(
        #     self.out_features, self.out_features, bias=False
        # )

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(adapter_name)

    def merge(
        self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        assert False, "Merging is not supported for Linear layers in SAE."

    def unmerge(self):
        assert False, "Unmerging is not supported for Linear layers in SAE."

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)


class Linear(nn.Module, TopKSaeLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str = None,
        k: int = 32,
        num_latents: int = 0,
        expansion_factor: int = 32,
        dead_tokens_threshold: int = 10000000,
        **kwargs,
    ):
        super().__init__()
        TopKSaeLayer.__init__(self, base_layer, **kwargs)

        self._activate_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            k=k,
            num_latents=num_latents,
            expansion_factor=expansion_factor,
        )
        self.num_tokens_fired = torch.zeros(self.num_latents, dtype=torch.int64)
        self.dead_tokens_threshold = dead_tokens_threshold
        # self.set_decoder_norm_to_unit_norm()

    def eager_decode(
        self, top_indices: torch.Tensor, top_acts: torch.Tensor, W_dec: torch.Tensor
    ):
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
        acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        return acts @ W_dec.mT


    def forward(self, x: torch.Tensor, sae_forward=False, *args: Any, **kwargs: Any) -> torch.Tensor:
        if sae_forward:
            return self.sae_forward(x, *args, **kwargs)
        
        self._check_forward_args(x, *args, **kwargs)
        result = self.base_layer(x, *args, **kwargs)
        final_result = result
        return final_result


    def sae_forward(self, result, return_acts=False) -> torch.Tensor:
        # result = x
        torch_result_dtype = result.dtype  
            # If we use sae, sae is reconstruction, we add and avg multiple saes
        final_result = torch.zeros_like(result, dtype=torch_result_dtype)
        logs = dict()
        if result.dim() == 3:
            num_tokens = result.shape[0] * result.shape[1]
        else:
            num_tokens = result.shape[0]

        for activate_adapter in self.active_adapters:
            encoder = self.sae_encoder[activate_adapter]
            W_dec = self.sae_W_dec[activate_adapter]
            bias = self.sae_b_dec[activate_adapter]
            k = self.k[activate_adapter]
            result = self._cast_input_dtype(result, encoder.weight.dtype)
            # Remove decoder bias as per Anthropic
            
            # print(f"debug: num_tokens={num_tokens} self.out_features={self.out_features} self.num_latents={self.num_latents} k={k}")
            # print(f"debug: result dtype={result.dtype}, shape={result.shape}")
            # print(f"debug: encoder weight dtype={encoder.weight.dtype}, shape={encoder.weight.shape}")
            # print(f"debug: W_dec dtype={W_dec.dtype}, shape={W_dec.shape}")
            # print(f"debug: bias dtype={bias.dtype}, shape={bias.shape}")
            
            sae_in = result - bias
            pre_act: torch.Tensor = torch.nn.functional.relu(encoder(sae_in))
            top_acts, top_indices = pre_act.topk(k, sorted=False)
            sae_out = self.eager_decode(top_indices, top_acts, W_dec.mT)
            sae_out = sae_out + bias
            for indice in top_indices:
                self.num_tokens_fired[indice.cpu()] += num_tokens
            final_result += sae_out
            
            if "pre_act" in logs.keys(): 
                logs["pre_act"].append(pre_act)
                logs["top_indices"].append(top_indices)
                logs["top_acts"].append(top_acts)
            else:
                logs["pre_act"] = [pre_act]
                logs["top_indices"] = [top_indices]
                logs["top_acts"] = [top_acts]

        final_result /= len(self.active_adapters)
        final_result = final_result.to(torch_result_dtype)
        # return 0
        # import pdb;pdb.set_trace()
        
        if return_acts:
            return final_result, logs
        
        return final_result

    
    def sae_mapping_forward(self, patches, tokens) -> torch.Tensor: 
        """Forward mapping linear on image patches and text tokens used for distance calculation.
        # no use
        Args:
            patches (Tensor): The image patches [n_patch, dim].
            tokens (Tensor): The text tokens [n_token, dim].
        """
        n_patch, dim = patches.shape
        n_token, dim = tokens.shape
        # for multiple sae, we avg emb
        token_emb = 0
        # sim = torch.zeros(1, dtype=patches.dtype, device=patches.device).sum()  # scalar
        for activate_adapter in self.active_adapters:
            # patch_map = self.sae_patch_map[activate_adapter]
            token_map = self.sae_token_map[activate_adapter]
            
            # patch_emb = patch_map(patches)
            token_emb = token_emb + token_map(tokens)
            
            # token_patch_sim = torch.einsum("td,pd->tp", token_emb, patches)
            # token_patch_sim = token_patch_sim * (dim ** -0.5)
            
            # sim + token_patch_sim

        token_emb = token_emb / len(self.active_adapters)
        # sim = sim / len(self.active_adapters)
        # sim = torch.softmax(sim, dim=-1)
            
        return token_emb
    
    
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        # import pdb; pdb.set_trace()
        assert self.sae_W_dec is not None, "Decoder weight was not initialized."
        for k in self.sae_W_dec.keys():
            eps = torch.finfo(self.sae_W_dec[k].dtype).eps
            norm = torch.norm(self.sae_W_dec[k].data, dim=1, keepdim=True)
            self.sae_W_dec[k].data /= norm + eps

    @property
    def dead_latent_percentage(self):
        percentage = sum(self.num_tokens_fired < self.dead_tokens_threshold) / len(
            self.num_tokens_fired
        )
        percentage = (
            percentage.item() if isinstance(percentage, torch.Tensor) else percentage
        )
        return percentage

    def __repr__(self):
        rep = super().__repr__()
        return "sae." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    topk_sae_config: TopKSaeConfig,
    **kwargs,
):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Linear):
        new_module = Linear(
            base_layer=target_base_layer,
            adapter_name=adapter_name,
            **kwargs,
        )

    return new_module
