import io
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers.utils import is_sagemaker_mp_enabled
from sklearn.cluster import AgglomerativeClustering
import wandb
import os

class SaeTrainer(Trainer):
    debug_verbose_count = 1
    compute_loss_count = -1

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        self.compute_loss_count += 1
        return self.compute_loss_instance(model, inputs, return_outputs, num_items_in_batch)

    def compute_loss_instance(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if (
            self.label_smoother is not None or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if "raw_images" in inputs:
            raw_images = inputs.pop("raw_images")
        else:
            raw_images = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        tokenizer = self.data_collator.tokenizer
        processor = self.data_collator.processor
        model_type = model.model_name      # manually added in train.py
        self.model_type = model_type
        patch_size = processor.image_processor.patch_size       # getattr(model.config, "patch_size", None)
        # if hasattr(model.config, "vision_config"):
        #     patch_size = getattr(model.config.vision_config, "patch_size", patch_size)
        
        model_outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        assert self.args.target_layer < len(model_outputs['hidden_states']) - 2     # not for the last layer
        input_hidden_states = model_outputs['hidden_states'][self.args.target_layer+1]
        if self.args.cluster_attn_strategy == "nextlayer":
            attentions = model_outputs['attentions'][self.args.target_layer+1].detach()  # +1 for next layer, since attentions do not have input shift.
        elif self.args.cluster_attn_strategy == "all":      # mean of all layers
            attentions = 0
            for attn in model_outputs['attentions']: 
                attentions = attentions + attn.detach()
            attentions = attentions / len(model_outputs['attentions'])
        else:
            raise NotImplementedError(f"Unknown cluster_attn_strategy {self.args.cluster_attn_strategy}")
            
        # print(f"attentions shape: {attentions.shape}, min {attentions.min().item()}, max {attentions.max().item()}, {attentions.flatten()[:5]}")     
        # [bs, num_attention_heads 28, n_patch 1630, n_patch 1630]   num_attention_heads divides hidden_states into heads 3584 / 28 = 128
        # apply 28 attention heads to focus on different region of hidden states. 
        # print(self.model.base_model.model.model.layers[self.args.target_layer].mlp.down_proj.sae_encoder['default'].weight)
        # print(self.model.base_model.model.model.layers[self.args.target_layer].mlp.down_proj.sae_W_dec['default'].norm(dim=1))
        # Possibly wrap up here, we use the original unwrapped model
        # output_hidden_dict = self.model.base_model.output_hidden_dict
        # input_hidden_dict = self.model.base_model.input_hidden_dict
        # pdb.set_trace
        per_layer_loss = {}
        # total_loss = 0
        # output_hidden_states, logs = self.model.base_model.model.model.layers[self.args.target_layer].mlp.down_proj.sae_forward(input_hidden_states, return_acts=True)
        output_hidden_states, logs = model.base_model.model.model.layers[self.args.target_layer].mlp.down_proj(input_hidden_states, sae_forward=True, return_acts=True)
        # logs: "pre_act" "top_indices" "top_acts"

        # import pdb; pdb.set_trace()
        e = output_hidden_states - input_hidden_states
        total_variance = (
            (input_hidden_states - input_hidden_states.mean(0).mean(0)).pow(2).sum()
        )
        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance
        per_layer_loss['fvu_loss'] = fvu.item()
        
        # Use fvu as the main loss
        total_loss = fvu
        
        outputs = {"input_hidden_states": input_hidden_states, 
                   "output_hidden_states": output_hidden_states,
                   "total_variance": total_variance,
                   "l2_loss": l2_loss,
                   "fvu": fvu}

        if self.debug_verbose_count > 0:
            print(f"model {model_type}: patch_size: {patch_size}")
            print("Inputs:")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    log_str = f"{key}: {tuple(value.shape)} min {value.min().item()}, max {value.max().item()}, {value.flatten()[:5]}"
                else:
                    log_str = f"{key}: {type(value)}"
                print(log_str)
            """ 
            model Qwen/Qwen2.5-VL-3B-Instruct: patch_size: 14
            Inputs:
            pixel_values: torch.Size([8092, 1176]) min -1.7890625, max 2.140625, tensor([-1.7812, -1.7812, -1.7812, -1.7812, -1.7812], device='cuda:0',                                                                                                                                                                                        
                dtype=torch.bfloat16)                                                                                                                                                                                                                                                                                                       
            image_grid_thw: torch.Size([3, 3]) min 1, max 86, tensor([ 1, 86, 64,  1, 32], device='cuda:0')                                                                                                                                                                                                                                    
            input_ids: torch.Size([3, 1630]) min 11, max 151655, tensor([151644,   8948,    198,   2610,    525], device='cuda:0')                                                                                                                                                                                                             
            attention_mask: torch.Size([3, 1630]) min False, max True, tensor([True, True, True, True, True], device='cuda:0')       
            """
            text = tokenizer.decode(inputs["input_ids"][0])      # , skip_special_tokens=True)
            compressed_text = text_compress(text, model_type=model_type)
            print(f"decode for [0]: \n{compressed_text}")
            print(f"merge_size: {getattr(processor.image_processor, 'merge_size', 1)}")
            """
            decode for [0]:                                                                                                                                                                                                                                                                                                                    
            <|im_start|>system                                                                                                                                                                                                                                                                                                                 
            You are a helpful assistant.<|im_end|>                                                                                                                                                                                                                                                                                             
            <|im_start|>user                                                                                                                                                                                                                                                                                                                   
            <|vision_start|><|image_pad|>*1376<|vision_end|>                                                                                                                                                                                                                                                                                   
            OCR this image section by section, from top to bottom, and left to right. Do not insert line breaks in the output text. If a word is split due to a line break in the image, use a space instead.<|im_end|>                                                                                                                        
            <|im_start|>assistant                                                                                                                                                                                                                                                                                                              
            e other two tu rrets abreast th e forward fun nel. The 8-in ch guns we re the Mark VI ty pe, and they fir ed s hell s at a mu zzl e v eloc ity of . T hey we re suppli ed with 125 shells per g un. The 6-inc h guns w ere placed i n casemates i n the hull. The 6-i nch Mark VI gu ns fired a shell at .For close-range defense a
            gai nst torpedo boats they carried tw elve 3-inch/50 caliber guns mou nted in casemates along the side of the hull and twelve 3-pou nder guns. As was stan dard for capital ships of the period the class carried four 21 inch torpe do tubes submerged in her hull on the br oadside. They were initially equipp<|im_end|>        

            merge_size: 2
            ## This first image 86 * 64 -> merge_size 2 -> 86*64/4 = 1376 which is <|image_pad|> size. 
            """
            print(
                f"input_hidden_states: {input_hidden_states.shape} min {input_hidden_states.min().item()}, max {input_hidden_states.max().item()}, {input_hidden_states.flatten()[:5]}, "
                f"\n"
                f"fvu loss: {fvu.item():.4f}, "
                f"l2_loss: {l2_loss.item():.4f}, "
                f"total_variance: {total_variance.item():.4f}"
                )
            """
            input_hidden_states: torch.Size([3, 1630, 2048]) min -2896.0, max 1624.0, tensor([ 4.5625,  2.5000,  3.1406, -1.2734, -0.4648], device='cuda:0',                                                                                                                                                                                   
                dtype=torch.bfloat16),                                                                                                                                                                                                                                                                                                      
            fvu loss: 1.2422, l2_loss: 114819072.0000, total_variance: 92274688.0000
            """
            for key, value in logs.items():
                print(f"{key}: len {len(value)} [0] {value[0].shape} min {value[0].min().item()}, max {value[0].max().item()}: {value[0].flatten()[:10]}")
            """
            pre_act: len 1 [0] torch.Size([3, 1630, 65536]) min 0.0, max 188.0: tensor([12.5000, 10.2500,  5.5938,  0.0000, 40.2500,  0.0000, 11.1250, 46.7500,                                                                                                                                                                                
                5.1875, 70.0000], device='cuda:0', dtype=torch.bfloat16, grad_fn=<SliceBackward0>)                                                                                                                                                                                                                                                                                                   
            top_indices: len 1 [0] torch.Size([3, 1630, 64]) min 5, max 65533: tensor([ 844, 1193, 1836, 2890, 3375, 4122, 4170, 4655, 5429, 5640],                                                                                                                                                                                            
                device='cuda:0')                                                                                                                                                                                                                                                                                                            
            top_acts: len 1 [0] torch.Size([3, 1630, 64]) min 3.09375, max 188.0: tensor([147., 154., 147., 148., 154., 153., 154., 150., 159., 159.],                                                                                                                                                                                         
                device='cuda:0', dtype=torch.bfloat16, grad_fn=<SliceBackward0>)
            NOTE: [0] because peft sae allows multiple adapters, here we only have one 'default' adapter.
            """
        
        pre_act = logs['pre_act'][0]
        top_indices = logs['top_indices'][0]
        
        # regularization
        sae_group_sparsity_coeff = getattr(self.args, "sae_group_sparsity_coeff", 0.0)
        if sae_group_sparsity_coeff > 0.0:
            # Add group sparsity reg
            batch_size = input_hidden_states.shape[0]
            per_layer_loss['gs'] = 0
            per_layer_loss['es'] = 0
            outputs['reg'] = 0
            
            # for each sample
            for sample_id in range(batch_size):
                token_infos = self.get_token_infos(
                    sample_id, input_hidden_states, inputs, tokenizer, processor, 
                    verbose=self.debug_verbose_count > 0)
                patch_indices = [idx for idx, token_info in enumerate(token_infos) if token_info["type"] == "image_patch"]
                text_indices = [idx for idx, token_info in enumerate(token_infos) if token_info["type"] == "text_token"]
                patch_acts = pre_act[sample_id, patch_indices, :]   # [n_img_patches, sae_dim]
                top_patch_indices = top_indices[sample_id, patch_indices, :]   # [n_img_patches, topk]
                text_acts = pre_act[sample_id, text_indices, :]      # [n_text_tokens, sae_dim]            
                top_text_indices = top_indices[sample_id, text_indices, :]   # [n_text_tokens, topk]
                image_patches = input_hidden_states[sample_id, patch_indices, :]   # [n_img_patches, dim]
                text_tokens = input_hidden_states[sample_id, text_indices, :]      # [n_text_tokens, dim]
                n_img_patches, dim = image_patches.shape
                n_text_tokens = text_tokens.shape[0]
                
                attn_weights = attentions[sample_id]    # [num_attention_heads, n_token, n_token]
                attn_weights_mean = attn_weights.mean(0)    # [n_token, n_token]
                image_attention = attn_weights_mean[patch_indices][:, patch_indices]       # [n_img_patches, n_img_patches]
                
                merge_size = getattr(processor.image_processor, 'merge_size', 1)   # 2 for qwenvl
                image_grid_thw = inputs['image_grid_thw']
                image_grid_thw = image_grid_thw[sample_id]
                image_grid_thw = image_grid_thw / merge_size
                cluster_labels, cluster_info = self.clustering(
                    image_attention, image_grid_thw=image_grid_thw, verbose=self.debug_verbose_count > 0
                )
                
                # binary trick on acts with top-k indices
                act_bin = torch.zeros_like(patch_acts)
                for p_id in range(patch_acts.shape[0]):
                    act_bin[p_id, top_patch_indices[p_id]] = 1.0
                    act_bin[p_id, patch_acts[p_id] == 0] = 0.0    # keep zero acts to be zero in act_bin
                    # This is useful if num of actived neurons is less than topk
                
                act_bin = (act_bin - patch_acts).detach() + patch_acts
                
                # compute group sparsity
                reg, group_sparsity_details = self.compute_group_sparsity(
                    cluster_labels, act_bin, verbose=self.debug_verbose_count > 0)
                per_layer_loss['gs'] += group_sparsity_details['gs'].item()
                per_layer_loss['es'] += group_sparsity_details['es'].item()
                outputs['reg'] += reg
                
            per_layer_loss['gs'] = per_layer_loss['gs'] / batch_size
            per_layer_loss['es'] = per_layer_loss['es'] / batch_size
            outputs['reg'] = outputs['reg'] / batch_size
            
            total_loss = total_loss + sae_group_sparsity_coeff * outputs['reg']
        
        # import pdb; pdb.set_trace()
        aux_log_info = self.model.base_model.get_aux_log_info()
        if aux_log_info:
            for key, value in aux_log_info.items():
                per_layer_loss[key] = value

        self.log(per_layer_loss)

        self.debug_verbose_count -= 1
        
        return (total_loss, outputs) if return_outputs else total_loss

    def clustering(self, attn_weights_mean, image_grid_thw, verbose=False):
        # attn_weights_mean: [n_patches, n_patches]
        # image_grid_thw [t, h, w] or [h, w]
        attentions = (attn_weights_mean + attn_weights_mean.T) / 2    # single-direction -> bi-direction
        attn_dist_matrix = attentions.max() - attentions 
        # normalize attn_dist_matrix to [0, 1] using its min and max to avoid scale issues
        min_val = attn_dist_matrix.min()
        max_val = attn_dist_matrix.max()
        range_val = max_val - min_val
        if range_val.abs() > 1e-12:
            attn_dist_matrix = (attentions.max() - attentions - min_val) / range_val
        else:
            # if all values are the same, produce a zero matrix
            attn_dist_matrix = torch.zeros_like(attn_dist_matrix)
        attn_dist_matrix = attn_dist_matrix.to(torch.float32).cpu().numpy()

        # add spatial information
        def get_patch_spatial_distance_matrix(grid_h: int, grid_w: int, mode: str = "manhattan", normalize: bool = False) -> np.ndarray:
            N = int(grid_h) * int(grid_w)
            if N == 0:
                return np.zeros((0, 0), dtype=np.float32)

            # row/col for each linear index (row-major)
            idx = np.arange(N)
            rows = idx // grid_w
            cols = idx % grid_w

            # broadcast compute difference matrices
            dr = np.abs(rows[:, None] - rows[None, :])
            dc = np.abs(cols[:, None] - cols[None, :])

            if mode == "manhattan":
                dist = dr + dc
            elif mode == "euclidean":
                dist = np.sqrt(dr.astype(np.float32) ** 2 + dc.astype(np.float32) ** 2)
            elif mode == "chebyshev":
                dist = np.maximum(dr, dc)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            dist = dist.astype(np.float32)

            if normalize:
                maxv = dist.max()
                if maxv > 0:
                    dist = dist / float(maxv)

            return dist

        spatial_diag_raw = get_patch_spatial_distance_matrix(image_grid_thw[-2].item(), image_grid_thw[-1].item(), normalize=True)  # [0, 1]
        spatial_diag = spatial_diag_raw ** getattr(self.args, "cluster_spatial_coeff", 0.02)
        distance_matrix = attn_dist_matrix * spatial_diag
        # distance_matrix = spatial_diag
        
        n_clusters = min(getattr(self.args, "sae_clustering_n_clusters", 20), attn_dist_matrix.shape[0])
        agnes = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        labels = agnes.fit_predict(distance_matrix)
        labels = torch.tensor(labels, device=attn_weights_mean.device)

        return labels, {'n_clusters': n_clusters, 'attn_dist_matrix': attn_dist_matrix, 'spatial_diag': spatial_diag_raw, 'distance_matrix': distance_matrix}

    def get_token_infos(self, sample_id, input_hidden_states, inputs, tokenizer, processor, verbose=False):
        batch_size, num_tokens, dim = input_hidden_states.shape
        merge_size = getattr(processor.image_processor, 'merge_size', 1)
        patch_size = processor.image_processor.patch_size
        
        # check patch position and distance
        image_grid_thw = inputs["image_grid_thw"]  # [batch_size, 3]
        # Locate the position of img patches within input_ids
        token_infos = get_token_info(
            model_type=self.model_type,
            num_tokens=num_tokens,
            input_ids=inputs['input_ids'][sample_id],
            tokenizer=tokenizer,
            image_grid_thw=image_grid_thw[sample_id],
            patch_size=patch_size,
            merge_size=merge_size,
            verbose=verbose,
        )
        return token_infos

    def compute_group_sparsity(self, labels, acts, verbose=False):
        # labels tensor[n_tokens,]
        # acts tensor[n_tokens, sae_dim]
        sae_dim = acts.shape[1]

        n_groups = labels.max().item() + 1
        # group_sparsity = torch.tensor(0.0, device=acts.device, dtype=acts.dtype)        # bfloat16
        group_sparsity_matrix = []
        num_tokens_in_groups = []
        for group_id in range(n_groups):
            group_mask = (labels == group_id)
            num_tokens_in_group = int(group_mask.sum().item())
            num_tokens_in_groups.append(num_tokens_in_group)
            if num_tokens_in_group == 0:         #  == 0 or  < 4; only reg on groups that more than 3 patches, to filter out small noisy groups
                group_sparsity_matrix.append(torch.zeros(sae_dim, device=acts.device, dtype=acts.dtype))
                continue
            
            group_acts = acts[group_mask]  # [n_tokens_each_group, sae_dim]
            group_norm = torch.sqrt((group_acts ** 2).mean(dim=0) + 1e-8)       # [sae_dim]
            group_sparsity_matrix.append(group_norm)
            
        group_sparsity_matrix = torch.stack(group_sparsity_matrix, dim=0)  # [n_groups, sae_dim]
        
        # compute gs according to group_sparsity_matrix
        gs = group_sparsity_matrix.mean(dim=1)    # l_1 norm [n_groups]
        gs = gs.mean() * 100        # since sae_dim is too large and sparse
        
        # compute es according to group_sparsity_matrix
        # es = group_sparsity_matrix.mean(dim=0)    # [sae_dim]
        es = group_sparsity_matrix.mean(dim=0)     # l_1 norm [sae_dim]
        es = 0.5 * (es ** 2).mean() * 1000        # since sae_dim is too large and sparse
        
        loss = gs + es
        
        return loss, {'group_sparsity': group_sparsity_matrix, 'n_tokens_in_groups': num_tokens_in_groups, 
                         'gs': gs, 'es': es}
    
def text_compress(text, model_type):
    # 统计相邻 '<|image_pad|>' 出现的次数，并压缩
    # 获取特殊 token id
    if 'qwen2.5-vl-' in model_type.lower(): 
        image_token = "<|image_pad|>"
        pad_token = "<|endoftext|>"
    elif 'llava' in model_type.lower():
        image_token = "<image>"
        pad_token ="<pad>"
    else:
        raise Exception(f"model_type {model_type} not supported yet for `text_compress`.")
        
    count = 0
    compressed = []
    i = 0
    while i < len(text):
        if text.startswith(image_token, i):
            cnt = 1
            j = i + len(image_token)
            while text.startswith(image_token, j):
                cnt += 1
                j += len(image_token)
            if cnt > 1:
                compressed.append(f"{image_token}*{cnt}")
            else:
                compressed.append(image_token)
            count += cnt
            i = j
        elif text.startswith(pad_token, i):
            cnt = 1
            j = i + len(pad_token)
            while text.startswith(pad_token, j):
                cnt += 1
                j += len(pad_token)
            if cnt > 1:
                compressed.append(f"{pad_token}*{cnt}")
            else:
                compressed.append(pad_token)
            count += cnt
            i = j
        else:
            compressed.append(text[i])
            i += 1
    compressed_text = ''.join(compressed)
    return compressed_text


def get_token_info(
    model_type: str,
    num_tokens,
    input_ids,
    tokenizer,
    image_grid_thw,
    token_idx: int = None,
    patch_size=14,
    merge_size=1,
    verbose=False,
):
    """
    For qwen tokenizer.special_tokens_map
        {'eos_token': '<|im_end|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': ['<|im_start|>',
        '<|im_end|>',
        '<|object_ref_start|>',
        '<|object_ref_end|>',
        '<|box_start|>',
        '<|box_end|>',
        '<|quad_start|>',
        '<|quad_end|>',
        '<|vision_start|>',
        '<|vision_end|>',
        '<|vision_pad|>',
        '<|image_pad|>',
        '<|video_pad|>']}
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    Picture 1: <|vision_start|>(<|image_pad|*n)<|vision_end|>Describe this image.<|im_end|>
    <|im_start|>assistant
    A black Honda motorcycle parked in front of a garage.<|im_end|>
    
    For llava tokenizer.special_tokens_map
        {'bos_token': '<|begin_of_text|>',
        'eos_token': '<|eot_id|>',
        'pad_token': '<pad>',
        'image_token': '<image>'}
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    <image>
    What is shown in this image?<|eot_id|>
    输入:
        token_idx: int, hidden state 的 token 索引. 
        input_ids: torch.Tensor, shape [seq_len]
        tokenizer: Qwen2.5-VL 的 tokenizer
        image_grid_thw: torch.Tensor, shape [3]，每张图片的 (t, h, w) or [2], (h, w)
        patch_size: int, patch 大小
        merge_size: int, merge 大小, 2 for qwen2.5-VL, default 1
    例子: 
        # 假设 input_ids, tokenizer, image_grid_thw 已准备好
        info = get_token_info(token_idx=42, input_ids=input_ids[0], 
                                tokenizer=tokenizer, image_grid_thw=image_grid_thw)
        print(info)
    返回:
        dict, 包含 token 类型、文本 token 索引 或 图片 patch 坐标
    """
    # 获取特殊 token id
    if 'qwen2.5-vl-' in model_type.lower(): 
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        conversation_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        conversation_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        special_tokens = tokenizer.special_tokens_map
    elif 'llava' in model_type.lower():
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        conversation_start_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
        conversation_end_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['pad_token'])
    
    # 计算该图片 patch 数量
    if image_grid_thw.shape[0] == 3:
        t, h, w = image_grid_thw
        t = t.item()
        h = h.item()
        w = w.item()
        num_patches = t * h * w // (merge_size ** 2)
    else:
        h, w = image_grid_thw[0].item(), image_grid_thw[1].item()
        num_patches = h * w // (merge_size ** 2)
        
    img_verbose, txt_verbose = 1, 1
    if token_idx is None:
        candidate_tokens = list(range(num_tokens))
    else:
        candidate_tokens = [token_idx]
        
    patch_info = []
    patch_start = -1
    for token_idx in candidate_tokens:
        # image
        if input_ids[token_idx] == image_token_id:
            if patch_start == -1:
                patch_start = token_idx
                
            patch_id = token_idx - patch_start
            # 计算 patch 在图片中的坐标
            grid_h = h // merge_size
            grid_w = w // merge_size
            gh = patch_id // grid_w
            gw = patch_id % grid_w
            row = gh * merge_size * patch_size
            col = gw * merge_size * patch_size
            patch_info.append({
                "type": "image_patch",
                "token_idx": token_idx, 
                "patch_id": patch_id,
                "patch_grid": (gh, gw),
                "image_size": (h * patch_size, w * patch_size),
                "patch_bbox": (col, row, col + merge_size * patch_size, row + merge_size * patch_size),
            })
            
            if verbose and img_verbose > 0: 
                img_verbose -= 1
                print(f"patch_id: {patch_id}, "
                      f"patch_grid: {(gh, gw)}, "
                      f"patch_bbox: {(col, row, col + merge_size * patch_size, row + merge_size * patch_size)}")
                """
                patch_id: 0, patch_grid: (0, 0), patch_bbox: (0, 0, 28, 28)
                """
        elif input_ids[token_idx] == pad_token_id:      # pad token
            continue
        else:       # text or special token
            text_token = tokenizer.decode([input_ids[token_idx]])
            patch_info.append({
                "type": "text_token",
                "token_idx": token_idx, 
                "input_id": input_ids[token_idx],
                "text_token": text_token,
            })
            
            if verbose and txt_verbose > 0: 
                txt_verbose -= 1
                print(f"text_index: {input_ids[token_idx]}, text_token: {text_token}")
                """
                text_index: 151644, text_token: <|im_start|>
                """
    
    return patch_info
