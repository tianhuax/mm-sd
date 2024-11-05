import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)
from tqdm import tqdm
import time
from qwen_vl_utils import process_vision_info
from utils import get_mismatches, get_parsed_args, run_model, run_model_with_assistant
import asyncio
from spd import (
    Generation
)

"""
Preliminary benchmarks for Qwen2-VL-7B-Instruct-AWQ on A100: 
Average time per input (ms): 5569.69
Average time per token (ms): 45.80

Preliminary benchmarks for Qwen2-VL-7B-Instruct-AWQ on A10G: 
Average time per input (ms): 6049.42
Average time per token (ms): 50.24

Preliminary benchmarks for Qwen2-VL-2B-Instruct-AWQ on A10G: 
Average time per input (ms): 4998.40
Average time per token (ms): 47.20

Preliminary benchmarks for speculative decoding on A10G: 
Average time per input (ms): 792.51
Average time per token (ms): 46.62

"""

GEN_LEN = 128

# initialize model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct-AWQ",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# load draft model for speculative decoding
assistant_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct-AWQ",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# # limit the number of image tokens or else it'll take a ton of vram
# min_pixels = 256*28*28
# max_pixels = 1280*28*28 

# very conservative settings to prevent OOM
# Standard resolution commonly used in vision models (224x224)
min_pixels = 224*224  # = 50,176 pixels
# Maximum resolution that balances quality and memory
max_pixels = 384*384  # = 147,456 pixels

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    min_pixels=min_pixels, 
    max_pixels=max_pixels,
    do_resize=True,
    do_rescale=True,
    do_normalize=True
)
assistant_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    min_pixels=min_pixels, 
    max_pixels=max_pixels,
    do_resize=True,
    do_rescale=True,
    do_normalize=True
)

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
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs

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

