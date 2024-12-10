# Miners

- Actively submit optimized checkpoints of the specified model or pipeline. No need for continuous operation; can wait
  for results after submission
- Use custom algorithms or tools to enhance model performance
- Aim to produce the most generically optimized version of the model

## Setup

1. Pick an [active contest](../README.md#active-contests) and clone the baseline repository. for example:
    ```bash
    git clone --depth 1 https://github.com/womboai/flux-schnell-edge-inference
    ```
2. Make your own repository on a git provider such as `GitHub` or `HuggingFace` to optimize in. Ensure the repository is
   public.
3. Edit the `src/pipeline.py` file to include any loading or inference optimizations, and commit when finished.
    - See [Proposals for Optimizations](#proposals-for-optimizations) for ideas.
    - Ensure the repository follows the [Submission Requirements](#submission-requirements).
4. Clone the EdgeMaxxing repository:
    ```bash
    git clone --depth 1 https://github.com/womboai/edge-maxxing
    cd edge-maxxing/miner
    ```
5. Install [pipx](https://pipx.pypa.io/stable/installation/)
6. Install `uv`:
    ```bash
    pipx ensurepath
    pipx install uv
    ```
7. Run the submission script and follow the interactive prompts:
    ```bash
    uv run submit_model \
    --netuid 39 \
    --subtensor.network finney \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey}
    ```
8. Optionally, benchmark your model locally before submitting
    ```bash
    pipx install huggingface-hub[cli,hf_transfer]
    uv run submit_model {existing args} --benchmarking.on
    ```
    - Requires `Ubuntu 22.04`
    - Ensure you are on the correct hardware for the contest, e.g. `NVIDIA GeForce RTX 4090`

## Submission Requirements

- Must be a public repository
- All code within the repository must be under `16MB`. Include large Huggingface files and models in the `models` array
  in the `pyproject.toml`.
- The size of the repository + all Huggingface models must be under `100GB`.
- The pipeline must load within `240s` on the target hardware.
- Must be offline (network is disabled during benchmarking).
- Submission code must not be obfuscated. Obfuscation libraries like `pyarmor` are **not** allowed.
- You are free to build on top of other miner's work; however, submitting copied submissions is **not** allowed and may
  result in your coldkey being blacklisted!

## Proposals for Optimizations

There are several effective techniques to explore when optimizing machine learning models for edge devices. Here are
some key approaches to consider:

1. **Knowledge Distillation**: Train a smaller, more efficient model to mimic a larger, more complex one. This technique
   is particularly useful for deploying models on devices with limited computational resources.

2. **Quantization**: Reduce the precision of the model's weights and activations, typically from 32-bit floating-point
   to 8-bit integers. This decreases memory usage and computational requirements, making it possible to run models on
   edge devices. Additionally, exploring low-precision representation for weights (e.g., using 8-bit integers) can
   reduce memory bandwidth usage for memory-bound models, even if the actual compute is done in higher precision (e.g.,
   32-bit).

3. **TensorRT and Hardware-Specific Optimizations**: Utilize NVIDIA's TensorRT to optimize deep learning models for
   inference on NVIDIA GPUs. This involves more than just layer fusion; it includes optimizing assembly, identifying
   prefetch opportunities, optimizing L2 memory allocation, writing specialized kernels, and performing graph
   optimizations. These techniques enhance performance and reduce latency by tailoring the model to the specific
   hardware configuration.

4. **Hyperparameter Tuning**: Optimize the configuration settings of the model to improve its performance. This can be
   done manually or through automated methods such as grid search or Bayesian optimization. While not a direct edge
   optimization, it is an essential step in the overall process of model optimization.

We encourage developers to explore these optimization techniques or develop other approaches to enhance model
performance and efficiency specifically for edge devices.
