# Assisted Generation Benchmarks

Example command:
```
python benchmark_vlm_open.py
```

See `get_parsed_args()` in `utils.py` for a list of flags.

# Getting Started

Running on python version `3.11.10`, use pyenv to install it. 
Then create a virtual environment and install the dependencies in the following order: 

1. `pip install uv`
2. `uv pip install autoawq`
3. `uv pip install -r requirements.txt`
4. `uv pip install flash-attn --no-build-isolation`


