import os

import datasets
import torch
import transformers
import wandb

from sae import get_peft_sae_model
from sae.launch.config import ModelArguments, SaeConfig, TrainingArguments
from sae.trainer import SaeTrainer
from sae.utils import hf_processor, hf_tokenizer
from sae.utils.datasets import CacheDataset
from sae.utils.factory import ModelFactory, SaeFactory
from transformers import TrainerCallback

try:
    if not os.environ.get("WANDB_API_KEY", None):
        wandb.login(key=os.environ.get("WANDB_API_KEY", None))
except Exception as e:
    pass

from transformers import TrainerCallback

class SetDecoderNormCallback(TrainerCallback):

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self, "my_trainer"):
            trainer = self.my_trainer
        else:
            raise RuntimeError("Trainer instance not found in callback.")
        target_layer = trainer.args.target_layer
        trainer.model.base_model.model.model.layers[target_layer].mlp.down_proj.set_decoder_norm_to_unit_norm()

def main():
    parser = transformers.HfArgumentParser(
        [TrainingArguments, SaeConfig, ModelArguments]
    )
    trainer_args, sae_config_, model_args = parser.parse_args_into_dataclasses()
    sae_type = sae_config_.sae_type

    sae_config = SaeFactory.create_sae_config(
        sae_type=sae_type,
        sae_config=sae_config_.__dict__,
    )

    print(f"trainer args: \n{vars(trainer_args)}")
    print(f"sae config: \n{vars(sae_config)}")
    print(f"model args: \n{vars(model_args)}")
    """
sae config: 
{'task_type': None, 'peft_type': <SaeType.TOPK_SAE: 'TOPK_SAE'>, 'auto_mapping': None, 
'base_model_name_or_path': None, 'revision': None, 'inference_mode': False, 
'expansion_factor': 32, 'num_latents': 0, 'k': 64, 'target_modules': 
'model.layers.28.mlp.down_proj', 'dead_tokens_threshold': 100000, 
'warm_start': True, 'warm_start_path': '/data3/sae_results/sae_qwen25vl3b_l28/checkpoint-5000'}
model args: 
{'model_path': 'Qwen/Qwen2.5-VL-3B-Instruct', 'attn_implementation': 'sdpa'}
    """

    model_kwargs = {
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch.bfloat16 if trainer_args.bf16 else torch.float32,
        "model_name": model_args.model_path,
    }

    processor = hf_processor(model_args.model_path)
    tokenizer = hf_tokenizer(model_args.model_path)

    model = ModelFactory.create_model(**model_kwargs)
    model_config = model.config
    # print(model)
    model = get_peft_sae_model(model, sae_config)
    model.config = model_config
    model.model_name = model_args.model_path
    model.print_trainable_parameters()
    
    # print the sae_b_dec shape and device for debugging with zero3
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        r = torch.distributed.get_rank()
    else:
        r = 0
    sae_model = model.base_model.model.model.layers[trainer_args.target_layer].mlp.down_proj
    for name in sae_model.sae_b_dec.keys():
        b = sae_model.sae_b_dec[name]
        print(f"rank {r}: sae_b_dec[{name}].shape={None if b is None else tuple(b.shape)}, device={None if b is None else b.device}")
    
    # dataset = datasets.load_dataset(
    #     trainer_args.dataset_path, split=trainer_args.split, name=trainer_args.subset
    # )
    # pdb.set_trace()
    dataset = datasets.load_dataset(
        "parquet",
        data_files=f"{trainer_args.dataset_path}/next_part_*.parquet",
        cache_dir=trainer_args.dataset_path
    )[trainer_args.split]

    sae_dataset = CacheDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        processor=processor,
        text_key=trainer_args.text_key,
        image_key=trainer_args.image_key,
        video_key=trainer_args.video_key,
        audio_key=trainer_args.audio_key,
    )

    callback = SetDecoderNormCallback()
    trainer = SaeTrainer(
        model=model,
        args=trainer_args,
        data_collator=sae_dataset.get_collator(),
        train_dataset=sae_dataset,
        callbacks=[callback],
    )
    # import pdb; pdb.set_trace()
    callback.my_trainer = trainer  
    trainer.train()


if __name__ == "__main__":
    import torch
    print('cuda_available=', torch.cuda.is_available())
    print('cuda_version=', torch.version.cuda)
    print('device_count=', torch.cuda.device_count())
    print('C.device_count=', torch._C._cuda_getDeviceCount())
    main()
