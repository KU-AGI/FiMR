import torch
from transformers import AutoModelForCausalLM, AutoConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor
from collections import OrderedDict
import os
import json
from typing import Dict
import yaml
from peft import get_peft_model, LoraConfig, TaskType
from collections import defaultdict
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint

def get_model(config):
    model_path = config.model.model_path
    cache_dir = config.model.cache_dir
    flash_attn = config.model.flash_attn
    model_dtype = torch.bfloat16 if config.model.model_dtype == 'bf16' else torch.float32
    mode = config.mode

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir)

    if flash_attn:
        model_config = AutoConfig.from_pretrained(model_path)
        model_config.language_config.use_flash_attention_2 = True
        model_config.language_config._attn_implementation = "flash_attention_2"

        
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=model_dtype, 
            config=model_config
        )

    else:
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=model_dtype, 
                device_map=None,   
            )

    if mode == 'train':
        if config.model.gradient_checkpoint:
            vl_gpt.language_model.config.use_cache = False
            vl_gpt.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        
        
    return vl_gpt, vl_chat_processor

def get_processor(config):
    model_path = config.model.model_path
    cache_dir = config.model.cache_dir
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir)
    return vl_chat_processor

def apply_lora(model, config):

    print("Applying LoRA... ✨")
    
    lora_config = LoraConfig(
        r=config.peft.r,
        lora_alpha=config.peft.lora_alpha,
        target_modules=config.peft.target_modules,
        lora_dropout=config.peft.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    peft_model.print_trainable_parameters()
    
    return peft_model

def get_lora_config(ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(ckpt_config_path, "r") as file:
        ckpt_config = yaml.safe_load(file)

    # Extract LoRA config
    lora_config = LoraConfig(
        r=ckpt_config["lora"].get("lora_rank"),
        lora_alpha=ckpt_config["lora"]["lora_alpha"],
        target_modules=ckpt_config["lora"]["target_modules"],
        lora_dropout=ckpt_config["lora"]["lora_dropout"],
        modules_to_save=ckpt_config["lora"].get("modules_to_save")                     
    )
 
    return lora_config
    
def load_deepspeed_ckpt(model, ckpt_path, dtype='bf16'):
    sft_state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)

    corrected_state_dict = {}
    keys_were_corrected = False

    prefix_to_remove = 'model.'

    print(f"Attempting to remove prefix: '{prefix_to_remove}'")

    for k, v in sft_state_dict.items():
        if k.startswith(prefix_to_remove):
            new_key = k[len(prefix_to_remove):] 
            corrected_state_dict[new_key] = v
            keys_were_corrected = True
        else:
            corrected_state_dict[k] = v 
            print(f"WARNING: Found a key without expected prefix: {k}")

    if keys_were_corrected:
        print(f"Successfully corrected '{prefix_to_remove}' prefix from SFT keys.")
    else:
        print("ERROR: No keys were corrected. The prefix 'model.' was not found.")

    print("Loading corrected state_dict into model (strict=False)...")
    load_result = model.load_state_dict(corrected_state_dict, strict=False)
    
    # 4. Verify load result.
    print("--- Load Result (after correction) ---")
    print(f"Missing Keys : {load_result.missing_keys}")
    print(f"Unexpected Keys : {load_result.unexpected_keys}")

    model_dtype = torch.bfloat16 if dtype == 'bf16' else torch.float32
    model = model.to(model_dtype)   

    return model