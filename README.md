<div align="center">

# WOMBO EdgeMaxxing Subnet: Optimizing AI Models for Consumer Devices, Enabling Millions to Contribute to a Decentralized Network

![WOMBO Cover](https://content.wombo.ai/bittensor/SN39_cover.jpeg "WOMBO AI")
#### *In the annals of the digital age, a grand saga unfolds. In a realm where the forces of artificial intelligence are harnessed by a select few, the question arises: shall this power remain concentrated, or shall it be distributed for the benefit of all humankind?*


</div>

# Table of Contents

- [About WOMBO](#about-wombo)
- [About w.ai](#about-wai)
- [Intro to EdgeMaxxing Subnet](#edgemaxxing-subnet)
- [Miners and Validators Functionality](#miners-and-validators)
    - [Incentive Mechanism and Reward Structure](#incentive-mechanism-and-reward-structure)
    - [Miners](#miners)
    - [Validators](#validators)
-  [Get Started with Mining or Validating](#running-miners-and-validators)
    - [Running a miner](#running-a-miner)
    - [Running a validator](#validator-setup)
- [Roadmap](#roadmap)

## About WOMBO
WOMBO is one of the world’s leading consumer AI companies, and early believers in generative AI.

We've launched - two #1 apps, [WOMBO](http://wombo.ai/) and [Dream](http://dream.ai/) which have been **downloaded over 200M times** and have **each** **hit #1 on the app stores in 100+ countries**

These results were **only possible due to the immense capabilities of bleeding edge generative AI techniques and the power of open source AI**. Our unique understanding of this research space through a consumer entertainment lens allows us to craft products people love to use and share.

We are at the very beginning of the Synthetic Media Revolution, which will completely transform how people create, consume, and distribute content. We're building the apps and infrastructure to power this change and bring AI entertainment potential to the masses.

## About w.ai

### Democratizing the Future of AI
w.ai envisions a future where artificial intelligence is decentralized, democratized, and accessible to everyone. This vision is embodied in a global supercomputer composed of individual user devices—laptops, gaming rigs, and smartphones. By harnessing the untapped potential of these devices, w.ai aims to create a vast decentralized network of computing power, democratizing access to the most advanced AI technologies. This approach will foster a thriving ecosystem of AI applications, driving innovation and ensuring the benefits of AI are shared by all of humanity.

## EdgeMaxxing Subnet

### What is the goal?
The EdgeMaxxing subnet aims to create the world's most optimized AI models for consumer devices, starting with Stable Diffusion XL on the NVIDIA GeForce RTX 4090. 

The subnet will expand to support optimization for various end devices, models, and modalities overtime.

### Key Benefits of Optimized Models:
Optimizing AI models is crucial to realizing a vision of decentralized AI.
- **Accessibility:** Enabling these advanced models to run on consumer devices, from smartphones to laptops, bringing AI capabilities to everyone.
- **Decentralization:** Allowing millions of users to contribute their computing power, rather than relying on a small number of powerful miners, creating a truly distributed AI network.

By optimizing popular models like LLAMA3 and Stable Diffusion, we transform idle computing resources into valuable contributors to a global AI network. This democratizes both AI usage and creation, offering earning opportunities to millions.

### Current Subnet Focus
- **Current GPU:** NVIDIA GeForce RTX 4090
- **Current Model:** `stablediffusionapi/newdream-sdxl-20`
- **Netuid:** 39

## Miners and Validators

### Incentive Mechanism and Reward Structure

The EdgeMaxxing subnet defines specific models, pipelines, and target hardware for optimization. Miners and validators collaborate in a daily competition to improve AI model performance on consumer devices.

Miners are incentivized through a ranking-based reward system. The more effective their optimization, the higher their potential earnings.

Validators receive rewards for their consistent operation and accurate scoring.


### Competition Structure
1. Miners submit optimized models
2. Validators score submissions
3. Contest runs daily at 12 PM New York time
4. Miners receive rewards based on their ranking


### Miners
- Actively submit optimized checkpoints of the specified model or pipeline. No need for continuous operation; can wait for results after submission
- Use custom algorithms or tools to enhance model performance
- Aim to produce the most generically optimized version of the model

### Validators
- Must run on the specified target hardware (e.g., NVIDIA GeForce RTX 4090, M2 MacBook)
- Collect all miner submissions daily
- Benchmark each submission against the baseline checkpoint
- Score models based on:
    - Speed improvements
    - Accuracy maintenance
    - Overall efficiency gains
- Select the best-performing model as the daily winner

## Running Miners and Validators

### Running a miner

To start working with a registered hotkey, clone the repository and install poetry
```bash
# Poetry
if [ "$USER" = "root" ]; then
  apt install pipx
else
  sudo apt install pipx
fi

pipx ensurepath
pipx install poetry

# Repository
git clone https://github.com/womboai/edge-optimization-subnet
cd edge-optimization-subnet
```

There is no need to manage venvs in any way, as poetry will handle that.

### Miner setup
1. Go to the miner directory after cloning the repository,
    ```bash
    cd miner
    ```
2. Clone the base model into a directory `model`
    ```bash
    git clone https://huggingface.co/stablediffusionapi/newdream-sdxl-20 model
    ```
3. Make your own repository on huggingface to optimize in
4. Edit the miner/miner.py file, specifically the [optimize](https://github.com/womboai/edge-optimization-subnet/blob/main/miner/miner/miner.py#L20) function
5. If you have methods of optimization that you do not wish to do through the aforementioned python function, optimize directly in the `model` directory
6. Submit the model, changing the options as necessary
    ```bash
    poetry run python miner/miner.py \
        --repository {huggingface-repository} \
        --netuid {netuid} \
        --subtensor.network finney \
        --wallet.name {wallet} \
        --wallet.hotkey {hotkey} \
        --logging.trace \
        --logging.debug
    ```
    Add `--no_optimizations` if you have not changed the `optimize` function, and `no_commit` if the optimizations are already in the repository, and you just want to make the submission.<br>
    Additionally, you can pass `--commit_message` to add a commit message to the commit made if `no_commit` is not passed.
7. Validators will collect your submission on 12PM New York time and tested in the remainder of the day

### Validator setup
All that is needed for a validator is running on the current contest's GPU with a registered hotkey;<br>

This assumes using PM2, feel free to adjust for anything else
```bash
cd validator

pm2 start poetry --name edge-validator --interpreter none -- \
    run python validator/validator.py \
    --netuid {netuid} \
    --subtensor.network {network} \
    --wallet.name {wallet} \
    --wallet.hotkey {hotkey} \
    --logging.trace \
    --logging.debug
```

## Roadmap

### Phase 1: Initial Optimization 
- [ ] Implement and optimize initial AI models on the NVIDIA GeForce RTX 4090
- [ ] Establish a robust telemetry and model benchmarking system
- [ ] Develop incentive mechanism for miners optimizing and distributing high-performance models

### Phase 2: Expansion to Multiple Modalities
- [ ] Expand optimization capabilities to support various models, end consumer devices, and modalities
- [ ] Integrate with WOMBO’s flagship products (Dream and WOMBO Me) to leverage optimized models

### Phase 3: Open Source Contributions
- [ ] Release open-source AI pipelines and APIs for optimized model usage
- [ ] Encourage third-party developers to build applications and services on top of the EdgeMaxxing subnet

<br>
<br>
<br>

<div align="center">
  <img src="https://content.wombo.ai/bittensor/logo.png" alt="WOMBO AI" width="100" style="margin-bottom: 10px;"/>
  <p>Connect with us on social media</p>
  <a href="https://twitter.com/wombo" style="margin-right: 10px;">
    <img src="https://content.wombo.ai/bittensor/twitter.png" alt="Twitter" width="20"/>
  </a>
  <a href="https://www.instagram.com/wombo.ai/">
    <img src="https://content.wombo.ai/bittensor/instagram.png" alt="Instagram" width="20"/>
  </a>
</div>
