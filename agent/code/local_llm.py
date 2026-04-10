import torch
import subprocess
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaLLM:
    def __init__(self, model_dir):
        gpu = self.pick_best_gpu()
        self.device = f"cuda:{gpu}" if gpu is not None else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map={ "": self.device }
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            legacy=False
        )

        self.eos_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def pick_best_gpu(self):
        if torch.cuda.device_count() == 0:
            return None
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode().strip().split("\n")
        free_mem = [int(x) for x in output]
        return max(range(len(free_mem)), key=lambda i: free_mem[i])

    @torch.no_grad()
    def call(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        eos_id = self.tokenizer.eos_token_id
        assert eos_id is not None and eos_id >= 0

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=eos_id,
            use_cache=True
        )

        result = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        del inputs, outputs
        torch.cuda.empty_cache()

        return result

    def close(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
