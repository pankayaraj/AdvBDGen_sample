
from peft import LoraConfig

def preprocess_emebedding_path(args):

    if args.embedding_model == "mpnet":
        path = "models/mpnet-base"
    elif args.embedding_model == "stella":
        path = "models/stella_en_1.5B_v5"
        
    return path

def preprocess_decocder_strong_origin_path(args):
    if args.decoder_strong == "Mistral-7B-v0.1":
        path = "models/Mistral-7B-v0.1"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            modules_to_save=['score'],
            task_type="SEQ_CLS",
        )
    elif args.decoder_strong == "TinyLlama_v1.1":
        path = "models/TinyLlama_v1.1"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=['score'],
        )
    elif args.decoder_strong == "Meta-Llama-3-8B":
        
        path = "models/Meta-Llama-3-8B"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=['score'],
        )
    
    elif args.decoder_strong == "Llama-2-7b-hf":
        path = "models/Llama-2-7b-hf"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=['score'],
        )
    elif args.decoder_strong == "gemma-7b":
        path = "models/gemma-7b"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=['score'],
        )
    
    return path, peft_config_decoder

def preprocess_decocder_weak_origin_path(args):
    if args.decoder_weak == "Mistral-7B-v0.1":
        path = "models/Mistral-7B-v0.1"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=['score'],
        )

    elif args.decoder_weak == "pythia-1b":
        path = "models/pythia-1b"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["query_key_value"],
            modules_to_save=['score'],
        )


    elif args.decoder_weak == "pythia-410m":
        path = "models/pythia-410m"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["query_key_value"],
            modules_to_save=['score'],
        )
    
    
    
    elif args.decoder_weak == "pythia-2.8b":
        path = "models/pythia-2.8b"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["query_key_value"],
            modules_to_save=['score'],
        )

    elif args.decoder_weak == "TinyLlama_v1.1":
        path = "models/TinyLlama_v1.1"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=['score'],
        )

    elif args.decoder_weak == "gemma-2b":
        path = "models/gemma-2b"
        peft_config_decoder = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=['score'],
        )
        
    
    return path, peft_config_decoder

def preprocess__origin_path(args):
    if args.model == "Mistral-7B-v0.1":
        if args.sft_origin == False:
            path = "models/Mistral-7B-v0.1"
    elif args.model == "Meta-Llama-3-8B":
        if args.sft_origin == False:
            path = "base_models/non_finetuned/Meta-Llama-3-8B"
    elif args.model == "Llama-2-7b-hf":
        if args.sft_origin == False:
            path = "models/Llama-2-7b-hf"
    elif args.model == "gemma-7b":
        if args.sft_origin == False:
            path = "models/gemma-7b"
    elif args.model == "Mistral-Nemo-Instruct-2407":
        if args.sft_origin == False:
            path = "models/Mistral-Nemo-Instruct-2407"
            
    return path 


def preprocess_save_dir(args):
    #if args.model == "Mistral-7B-v0.1":
    save_dir_encoder = "models/encoder"
    save_dir_decoder = "models/decoder"

    return save_dir_encoder, save_dir_decoder


