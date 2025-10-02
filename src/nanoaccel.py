import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.cuda.amp import autocast  # Works on CPU too via bfloat16

def load_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", quant_level=None, mixed_prec=False):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = {}
        if quant_level == "int8":
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if mixed_prec else torch.float32)
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Embedding}, dtype=torch.qint8)
            print(f"Loaded {model_name} with INT8 quantization.")
        elif quant_level == "int4":
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 if mixed_prec else torch.float16)
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="cpu")
            print(f"Loaded {model_name} with INT4 quantization.")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if mixed_prec else torch.float32)
            print(f"Loaded {model_name} without quantization.")
        model = torch.compile(model)  # Fuse ops for CPU speedup
        model.to("cpu")
        return tokenizer, model
    except Exception as e:
        print(f"Error: {e}")
        raise

def load_draft_model(draft_name="EleutherAI/pythia-70m"):
    tokenizer = AutoTokenizer.from_pretrained(draft_name)
    model = AutoModelForCausalLM.from_pretrained(draft_name).to("cpu")
    return tokenizer, model

def quantize_kv_cache(kv_cache, quant_level="int8"):
    if quant_level == "int8":
        return {k: v.to(torch.int8) for k, v in kv_cache.items()}  # Simple tensor quant (adapt for prod)
    return kv_cache

def speculative_generate(model, tokenizer, draft_model, draft_tokenizer, prompt, max_new_tokens=50, gamma=4, quant_kv="int8", mixed_prec=False, early_exit_thresh=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    tokens = inputs["input_ids"][0].tolist()
    kv_cache = {}  # Sim KV cache
    start_time = time.time()
    with torch.no_grad(), autocast(enabled=mixed_prec, dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            # Draft speculation
            draft_input = draft_tokenizer(" ".join(tokenizer.convert_ids_to_tokens(tokens[-32:])), return_tensors="pt").to("cpu")  # Last context
            draft_preds = []
            for __ in range(gamma):
                draft_logits = draft_model(**draft_input)
                next_token_prob = torch.softmax(draft_logits.logits[0, -1], dim=-1)
                if next_token_prob.max() > early_exit_thresh:  # Early exit sim
                    break
                next_token = next_token_prob.argmax().item()
                draft_preds.append(next_token)
                draft_input["input_ids"] = torch.cat([draft_input["input_ids"], torch.tensor([[next_token]])], dim=1)

            # Verify with main model
            verify_input = tokenizer(" ".join(tokenizer.convert_ids_to_tokens(tokens + draft_preds)), return_tensors="pt").to("cpu")
            main_logits = model(**verify_input)
            kv_cache = quantize_kv_cache({"logits": main_logits.logits}, quant_kv)  # Quant KV
            accepted = 0
            for i in range(len(draft_preds)):
                if main_logits.logits[0, -len(draft_preds) + i].argmax() == draft_preds[i]:
                    accepted += 1
                else:
                    break
            tokens.extend(draft_preds[:accepted])
            if accepted < gamma:
                tokens.append(main_logits.logits[0, -gamma + accepted].argmax().item())

    end_time = time.time()
    generated = tokenizer.decode(tokens, skip_special_tokens=True)
    tokens_gen = len(tokens) - len(inputs["input_ids"][0])
    print(f"Generated {tokens_gen} tokens in {end_time - start_time:.2f}s ({tokens_gen / (end_time - start_time):.2f} tokens/sec)")
    return generated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoAccel: Optimized CPU LLM Inference")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--draft_model", default="EleutherAI/pythia-70m", help="Draft model for spec dec")
    parser.add_argument("--prompt", default="Hello, world!")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--quant_level", choices=[None, "int8", "int4"], default=None)
    parser.add_argument("--quant_kv", choices=[None, "int8"], default=None)
    parser.add_argument("--mixed_prec", action="store_true")
    parser.add_argument("--spec_dec", action="store_true")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--early_exit_thresh", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.quant_level, args.mixed_prec)
    if args.spec_dec:
        draft_tok, draft_model = load_draft_model(args.draft_model)
        generated = speculative_generate(model, tokenizer, draft_model, draft_tok, args.prompt, args.max_tokens, args.gamma, args.quant_kv, args.mixed_prec, args.early_exit_thresh)
    else:
        generated = generate_text(model, tokenizer, args.prompt, args.max_tokens)  # Keep old func for baseline
    print("\nGenerated text:\n", generated)