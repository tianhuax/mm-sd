import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoProcessor, 
    LlavaOnevisionForConditionalGeneration
)
from tqdm import tqdm
import time
from utils import get_mismatches, get_parsed_args, run_model, run_model_with_assistant
import asyncio
from spd_ov import (
    Generation
)

GEN_LEN = 128

# TODO: need to quantize these models if no space. Might need to run on L40 for additional VRAM 
# idk if both enabling both flash-attn2 and bits and bytes is compatible. 
# initialize model and processor
target_model_id = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    target_model_id,   
    torch_dtype=torch.float16, 
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True, 
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(target_model_id)

# load draft model for speculative decoding
assistant_model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
assistant_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    assistant_model_id,   
    torch_dtype=torch.float16, 
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True, 
    device_map="auto",
)
# TODO: figure out what other parameters are accepted by this processor. 
"""
In Qwen2-VL at least, these were additional parameters: 
    min_pixels=min_pixels, 
    max_pixels=max_pixels,
    do_resize=True,
    do_rescale=True,
    do_normalize=True
"""
assistant_processor = AutoProcessor.from_pretrained(assistant_model_id)


# # limit the number of image tokens or else it'll take a ton of vram
# min_pixels = 256*28*28
# max_pixels = 1280*28*28 

# very conservative settings to prevent OOM
# Standard resolution commonly used in vision models (224x224)
# TODO: figure out how this model's processor tokenizes images and whether that can be controlled
min_pixels = 224*224  # = 50,176 pixels
# Maximum resolution that balances quality and memory
max_pixels = 384*384  # = 147,456 pixels

num_samples = 3
temp = None # baseline with greedy sampling strategy to get quality guarantees
outputs = []
gen_time = []
num_tokens = []

# model generation kwargs
# TODO: port the huggingface source code to make this work. 
# relevant code can be found in transformers/generation/utils.py
generate_kwargs = {
    "max_new_tokens": GEN_LEN,
    "use_cache": True,
    # "assistant_model": assistant_model,
    # "tokenizer": processor,
    # "assistant_tokenizer": assistant_processor,
}
if temp is not None:
    generate_kwargs.update({
        "do_sample": True,
        "temperature": temp,
        "top_p": 0.001,
        "top_k": 1,
    })
else:
    generate_kwargs.update({
        "do_sample": False,
    })

spd = Generation(model, assistant_model, processor, generate_kwargs)

# FIXME: sanity check lmao
# spd = Generation(assistant_model, assistant_model, processor, generate_kwargs)

def process_image(image):
    messages = [
        {
        "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image"},
                ],
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process inputs but keep image_sizes on CPU
    inputs = processor(images=image, text=text, return_tensors='pt')
    
    return inputs.to(dtype=torch.float16, device='cuda')

def custom_collate_fn(batch):
    return [b["image"] for b in batch]


def main():


    ds = load_dataset("sayakpaul/coco-30-val-2014", split="train")
    # TODO: study batch inference later, this is a different setting and isn't within scope for now
    loader = DataLoader(
        ds, 
        batch_size=1, 
        collate_fn=custom_collate_fn
    )

    for i, image in tqdm(enumerate(loader), total=num_samples):
        if i >= num_samples:
            break

        # process image and prepare inputs
        # NOTE: uses image_sizes instead of image_grid_thw
        inputs = process_image(image[0])

        # run decoder generation
        start = time.time()
        generated_ids = spd.generate(inputs)

        # NOTE: using profiler for generation
        # generated_ids = spd.generate_with_profiling(inputs) # profile torch generation to identify bottlenecks
        end = time.time()

        # process output to be human readable
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs.append(output_text)

        # warmup GPU by discarding first two iterations from collected metrics
        if i >= 2:  
            gen_time.append(end - start)
            num_tokens.append(generated_ids.shape[1] - inputs.input_ids.shape[1])

    # print collected metric 
    print(f"outputs: {outputs}")
    print(f"Average time per input (ms): {(sum(gen_time) / len(gen_time))*1000:.2f}")
    print(f"Average time per token (ms): {(sum(gen_time) / sum(num_tokens))*1000:.2f}")
    print(f"Number of tokens generated: {sum(num_tokens)}")



if __name__ == "__main__":

    main()


    # args = get_parsed_args()

    # new_outputs = run_model_with_assistant(args, AutoTokenizer, AutoModelForCausalLM, run_prediction_loop)
    # og_outputs = run_model(args, AutoTokenizer, AutoModelForCausalLM, run_prediction_loop)

    # for i in range(len(og_outputs)):
    #     print("\nOG :", og_outputs[i])
    #     print("NEW:", new_outputs[i])

    # if args.temperature is None:
    #     get_mismatches(og_outputs, new_outputs, args.dtype)

