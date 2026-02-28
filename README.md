# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from HuggingFace Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/hexagon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, such as [ModelScope](https://modelscope.cn/), by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. For example, you may opt to downloading model checkpoints from ModelScope or other model sharing communities by setting the environment variable, e.g. `MODEL_ENDPOINT=https://www.modelscope.cn/`.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain


# Self-Speculative Decoding for DeepSeek MoE Models in llama.cpp

## Abstract

We implement self-speculative decoding for Mixture-of-Experts (MoE) language models within llama.cpp, exploiting the architectural separation between shared and routed experts in the DeepSeek model family. In standard speculative decoding, a smaller draft model proposes candidate tokens verified by the full model. Self-speculative decoding eliminates the need for a separate draft model by using a degraded version of the same model: specifically, we execute only the shared expert FFN while skipping routed experts during the draft phase, then verify with the full MoE computation. We describe the implementation, which modifies the llama.cpp graph construction layer to conditionally bypass routed expert computation, expose a public C API (`llama_set_moe_draft_mode`), and provide a standalone benchmark harness. Preliminary results on DeepSeek-Coder-V2-Lite (15.7B, IQ3_XXS) running on an NVIDIA RTX 5050 Laptop GPU (8 GB, Blackwell sm_120) show 35.4% draft acceptance rate at draft length 2, with effective throughput of 22.2 t/s against a 59.2 t/s baseline. Analysis reveals that V2-Lite's unusually large shared-to-routed parameter ratio (~2.6:1 per layer) renders the technique counterproductive on this model: the draft path performs 72% of full inference compute while producing low-quality predictions. We argue the technique is architecturally better suited to DeepSeek-V3/R1 (671B), where 8 of 256 routed experts dominate per-layer FLOPS and the shared expert is proportionally small.

## 1. Background

### 1.1 Speculative Decoding

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) accelerates autoregressive inference by using a fast draft model to propose *n* candidate tokens, then verifying the entire sequence in a single forward pass of the target model. Accepted tokens are kept; the first rejected token is replaced with the target model's prediction. The technique yields speedups proportional to the draft acceptance rate and the speed ratio between draft and target models.

### 1.2 Self-Speculative Decoding

Self-speculative decoding (Zhang et al., 2024, "Draft & Verify") eliminates the separate draft model by degrading the target model itself. Common approaches include skipping intermediate layers or, in MoE architectures, reducing the expert computation. The key advantage is zero additional memory overhead — no second model needs to be loaded.

### 1.3 DeepSeek MoE Architecture

DeepSeek's MoE models (V2, V2-Lite, V3, R1) compute each transformer layer's FFN output as:

```
FFN_out = SharedExpert(x) + RoutedExperts(x)
```

where `SharedExpert` is a standard dense FFN applied to every token, and `RoutedExperts` selects `k` of `N` expert FFNs via a learned gating function. The shared expert provides a "baseline" representation; routed experts add token-specific specialization.

| Model | Params | Experts (N) | Active (k) | Shared | Shared FFN dim | Expert FFN dim |
|-------|--------|-------------|------------|--------|----------------|----------------|
| V2-Lite | 15.7B | 64 | 6 | 2 | 10944 | 1408 |
| V2 | 236B | 160 | 6 | 2 | — | — |
| V3 / R1 | 671B | 256 | 8 | 1 | 2048 | 2048 |

The critical architectural observation: in V2-Lite, each shared expert FFN is large (dim 10944) relative to each routed expert (dim 1408). Two shared experts process every token, totaling ~134M multiply-accumulate operations per layer, versus ~52M for the 6 active routed experts. The shared path already dominates compute.

In V3/R1, this ratio inverts: routed experts collectively dominate per-layer FLOPS, making the shared-expert-only draft path significantly cheaper than full inference.

## 2. Implementation

### 2.1 Overview

The implementation modifies 7 files in llama.cpp (based on commit `d91ca639`), adding approximately 60 lines of code across the graph construction layer, context management, and public API.

### 2.2 API

```c
// include/llama.h
LLAMA_API void llama_set_moe_draft_mode(struct llama_context * ctx, int32_t n_expert);
```

Where `n_expert`:
- `-1` = normal mode (use all configured experts)
- `0` = skip all routed experts (shared expert only)  
- `1+` = use top-N routed experts (reduced MoE) — *currently limited by ggml tensor allocation; see Section 4*

### 2.3 Flag Propagation Chain

```
llama_set_moe_draft_mode(ctx, 0)              // Public C API
  → ctx->set_moe_draft_mode(0)                // llama_context method
    → moe_draft_n_expert = 0                  // llama_context member
      → graph_params(): moe_draft_n_expert    // Propagated to graph params
        → llm_graph_context::moe_draft_n_expert   // Available during graph build
          → if (moe_draft_n_expert != 0) {    // deepseek2.cpp / deepseek.cpp
                build_moe_ffn(...)            //   Full or reduced MoE
             } else {
                cur = ffn_shexp;              //   Shared expert only
             }
```

### 2.4 Graph Layer Surgery (deepseek2.cpp)

The core modification reorders the MoE computation to always compute the shared expert first, then conditionally add routed experts:

```cpp
// FFN shared expert (always computed)
ggml_tensor * ffn_shexp = build_ffn(cur, ...shared weights...);

if (moe_draft_n_expert != 0) {
    // Full or reduced MoE
    const int64_t n_expert_act = (moe_draft_n_expert > 0)
        ? (int64_t) moe_draft_n_expert : n_expert_used;
    ggml_tensor * moe_out = build_moe_ffn(cur, ..., n_expert, n_expert_act, ...);
    cur = ggml_add(ctx0, moe_out, ffn_shexp);
} else {
    // Draft mode: shared expert only
    cur = ffn_shexp;
}
```

### 2.5 Graph Reuse Safety

The `allow_reuse()` function in `llm_graph_params` includes `moe_draft_n_expert` in its topology comparison, ensuring the computation graph is rebuilt when switching between draft and verify modes:

```cpp
bool allow_reuse(const llm_graph_params & other) const {
    // ... existing checks ...
    return
        cparams.embeddings  == other.cparams.embeddings  &&
        cparams.causal_attn == other.cparams.causal_attn &&
        moe_draft_n_expert  == other.moe_draft_n_expert  &&  // NEW
        arch  == other.arch  && ...
}
```

### 2.6 Self-Speculative Loop

The benchmark harness (`moe-self-spec-test.cpp`) implements the standard speculative decoding loop:

1. **Draft phase**: Set `llama_set_moe_draft_mode(ctx, 0)`, generate `n_draft` tokens autoregressively using shared expert only
2. **KV rollback**: Remove draft-phase KV entries via `llama_memory_seq_rm()`
3. **Verify phase**: Set `llama_set_moe_draft_mode(ctx, -1)`, evaluate all draft tokens in a single batch with full MoE
4. **Accept/reject**: Greedily compare draft predictions against verify predictions; accept matching prefix, replace first mismatch with verify token
5. **KV trim**: Remove KV entries beyond accepted tokens

## 3. Preliminary Results

### 3.1 Hardware

- **GPU**: NVIDIA GeForce RTX 5050 Laptop GPU (8 GB GDDR7, Blackwell sm_120)
- **CUDA**: 12.8.61
- **Build**: llama.cpp b8184, native sm_120 compilation

### 3.2 Model

- **DeepSeek-Coder-V2-Lite-Instruct** (15.7B total, ~2.4B active)
- **Quantization**: IQ3_XXS (6.96 GB on disk, 3.06 bpw)
- **Context**: 512 tokens, full GPU offload (28/28 layers)

### 3.3 Benchmark: "Write a short story about a robot" (64 tokens)

| Configuration | Tokens | Acceptance | Draft t/s | Effective t/s | Time (ms) |
|---------------|--------|------------|-----------|---------------|-----------|
| Baseline (no speculation) | 64 | — | — | **59.2** | 1,081 |
| Self-spec, d=5, e=0 | 64 | 15.6% | 106.4 | 18.6 | 3,437 |
| Self-spec, d=2, e=0 | 64 | 35.4% | 61.6 | 22.2 | 2,884 |

### 3.4 Analysis

The results are a clear negative for V2-Lite. Two factors explain the failure:

**Factor 1: Draft speed ≈ Baseline speed.** At d=2, draft throughput is 61.6 t/s versus 59.2 t/s baseline — a mere 4% speedup from skipping routed experts. This confirms that on V2-Lite, the shared expert FFN (dim 10944 × 2 experts) dominates per-layer compute, while the routed experts (dim 1408 × 6 active) are already cheap. Skipping the cheap part doesn't help.

**Factor 2: Low acceptance rate.** At 35.4% (d=2) and 15.6% (d=5), the shared expert alone is a poor predictor of full-model behavior. The routed experts carry substantial semantic information on this model, meaning the draft deviates frequently from the verify path.

The combination is fatal: negligible draft speedup × low acceptance rate × verify overhead (full re-evaluation of draft tokens) = net slowdown.

**Shorter drafts help acceptance but not throughput.** Reducing draft length from 5 to 2 roughly doubles acceptance rate (15.6% → 35.4%) by reducing the probability of encountering a mismatch. However, the fundamental speed ratio problem remains.

### 3.5 Theoretical Speedup Model

For self-speculative decoding to yield a net speedup, the following inequality must hold:

```
(1 + α·d) / (d·t_draft + t_verify) > 1 / t_base
```

where `α` = acceptance rate, `d` = draft length, `t_draft` = time per draft token, `t_verify` = time to verify the batch, and `t_base` = time per baseline token. On V2-Lite with d=2: `t_draft ≈ t_base` (no draft speedup), making the inequality impossible to satisfy regardless of acceptance rate. The technique requires `t_draft << t_base`, which only holds when routed experts dominate compute.

## 4. Limitations and Known Issues

### 4.1 Top-N Expert Mode (n_expert > 0)

The `-e N` parameter for using top-N routed experts (instead of skipping all) is implemented in the graph builder but currently crashes at runtime. The ggml tensor allocator pre-allocates view buffers sized for the full expert count during graph reservation. When `build_moe_ffn` is called with a reduced `n_expert_used`, the resulting tensor views violate size assertions:

```
GGML_ASSERT(view_src == NULL || data_size == 0 || 
            data_size + view_offs <= ggml_nbytes(view_src)) failed
```

Fixing this requires either: (a) reserving buffers for the maximum expert count regardless of draft mode, which requires ggml-level changes to decouple reservation sizing from graph construction; or (b) using two separate contexts with different reservations for draft and verify modes.

### 4.2 Scheduler Overhead

Each transition between draft and verify mode invalidates the cached computation graph (via `allow_reuse()` topology mismatch), forcing a graph rebuild. On GPUs with CUDA graph support, this also invalidates the CUDA graph, triggering a warmup pass. For the n_expert=0 mode, the overhead is manageable since graph topology changes only in the presence/absence of MoE subgraphs. Future optimization could maintain two pre-built graphs and switch between them.

### 4.3 KV Cache Contamination

Draft-phase KV entries are computed with shared-expert-only representations, which differ from full-MoE representations. The current implementation rolls back all draft KV entries before verification (`llama_memory_seq_rm`), then recomputes them with full MoE. An alternative approach — keeping draft KV entries and only recomputing mismatched positions — would be incorrect because even "accepted" tokens have subtly different internal representations when routed experts are included.

## 5. Future Work

### 5.1 DeepSeek-V3 / R1 Evaluation (Primary)

The technique is architecturally better suited to DeepSeek-V3 (671B) where routed experts dominate per-layer compute. With 256 experts and 8 active per token, each with FFN dim 2048, the routed computation is ~8× the shared expert cost. Skipping routed experts would reduce draft compute to ~11% of full inference — a much more favorable speed ratio than V2-Lite's ~72%.

**Planned hardware**: 8× NVIDIA V100 (32 GB each, full NVLink), 256 GB total VRAM. Target model: DeepSeek-V3-0324 at IQ2_XS quantization (~180–200 GB).

### 5.2 Layer-Selective MoE Skipping

Instead of skipping routed experts in all layers, skip them only in middle layers (which empirically contribute less to output quality) while retaining full MoE in the first and last N layers. This approach: (a) preserves graph topology per-layer (no ggml allocation issues), (b) produces better draft quality by retaining experts where they matter most, and (c) is configurable per deployment.

### 5.3 Adaptive Draft Length

Dynamically adjust `n_draft` based on running acceptance rate. High acceptance → increase draft length to amortize verify cost over more tokens. Low acceptance → reduce draft length to minimize wasted computation.

### 5.4 Top-K Expert Draft Mode

Fix the ggml tensor allocation issue to enable top-1 or top-2 expert draft mode. Using the single highest-weighted expert (instead of all 6–8) preserves ~83–87% of compute savings while producing substantially better draft predictions than shared-expert-only mode.

### 5.5 Acceptance Rate vs. Quantization

Measure acceptance rate across quantization levels (IQ2, IQ3, Q4, Q6, Q8, F16) for the same model. Hypothesis: heavier quantization adds noise to routed expert contributions, effectively reducing the signal that distinguishes full-MoE from shared-only, thereby *increasing* acceptance rate. If confirmed, this would mean self-speculative decoding becomes more effective at lower quantization — a useful result for consumer hardware.

## 6. Build and Usage

### 6.1 Requirements

- llama.cpp (commit d91ca639 or later from this fork)
- CUDA Toolkit 12.8+ (for Blackwell sm_120) or 11.0+ (for V100 sm_70)
- CMake 3.21+
- C++17 compiler

### 6.2 Build

```bash
# Clone this fork
git clone https://github.com/<username>/llama.cpp.git
cd llama.cpp

# Configure (adjust -DCMAKE_CUDA_ARCHITECTURES for your GPU)
# sm_120 = RTX 5050/5060/5070/5080/5090 (Blackwell)
# sm_70  = V100
# sm_89  = RTX 4090
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120

# Build everything
cmake --build build --config Release

# Or build just the benchmark
cmake --build build --config Release --target moe-self-spec-test
```

### 6.3 Run Benchmark

```bash
# Shared expert only (e=0), draft length 2, with baseline comparison
moe-self-spec-test -m model.gguf -c 512 -ngl 999 -n 128 -d 2 -e 0 --baseline

# Options:
#   -m   model path (required)
#   -p   prompt (default: "Write a short story about a robot.")
#   -n   tokens to generate (default: 128)
#   -d   draft tokens per step (default: 5)
#   -e   draft expert count: 0=shared only (default: 0)
#   -c   context size (default: 512)
#   -ngl GPU layers to offload (default: 999)
#   --baseline  also run standard autoregressive for comparison
```

### 6.4 Use the API

```c
#include "llama.h"

// During speculative loop:

// Draft phase — shared expert only
llama_set_moe_draft_mode(ctx, 0);
// ... generate draft tokens ...

// Verify phase — full MoE
llama_set_moe_draft_mode(ctx, -1);
// ... verify batch ...
```

## 7. Files Modified

| File | Changes |
|------|---------|
| `include/llama.h` | `llama_set_moe_draft_mode()` declaration |
| `src/llama-graph.h` | `moe_draft_n_expert` in params struct and graph context; `allow_reuse()` topology check |
| `src/llama-graph.cpp` | Constructor initialization of `moe_draft_n_expert` |
| `src/llama-context.h` | `set_moe_draft_mode()` method; `moe_draft_n_expert` member |
| `src/llama-context.cpp` | Method implementation; `graph_params()` propagation; C API wrapper |
| `src/models/deepseek2.cpp` | Conditional MoE bypass in DeepSeek V2/V2-Lite/V3/R1 graph builder |
| `src/models/deepseek.cpp` | Same for DeepSeek V1 |
| `examples/moe-self-spec-test/` | Standalone benchmark harness |

## References

- Chen, C., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318.
- Leviathan, Y., et al. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
- Zhang, J., et al. (2024). "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding." ACL 2024.
- DeepSeek-AI. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv:2405.04434.
- DeepSeek-AI. (2025). "DeepSeek-V3 Technical Report." arXiv:2412.19437.

## License

This implementation is a patch against llama.cpp and is released under the same license (MIT).

```
--- a/include/llama.h
+++ b/include/llama.h
@@ -963,6 +963,12 @@
     // If true, all model tensors are activated during llama_decode() to load and cache their weights.
     LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup);
 
+    // Set MoE draft mode for self-speculative decoding.
+    //   n_expert = -1: normal mode (use all configured experts)
+    //   n_expert =  0: skip all routed experts (shared expert only)
+    //   n_expert = 1+: use top-N routed experts (reduced MoE)
+    LLAMA_API void llama_set_moe_draft_mode(struct llama_context * ctx, int32_t n_expert);
+
     // Set abort callback
     LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);
 
--- a/src/llama-graph.h
+++ b/src/llama-graph.h
@@ -553,6 +553,12 @@
 
     uint32_t n_outputs;
 
+    // self-speculative decoding: override n_expert_used in MoE layers
+    //   -1 = normal mode (use hparams.n_expert_used)
+    //    0 = skip all routed experts (shared expert only)
+    //   1+ = use top-N routed experts (draft with reduced expert count)
+    int32_t moe_draft_n_expert = -1;
+
     llm_graph_cb cb;
 
     llm_graph_result * res;
@@ -615,6 +621,7 @@
         return
             cparams.embeddings  == other.cparams.embeddings  &&
             cparams.causal_attn == other.cparams.causal_attn &&
+            moe_draft_n_expert  == other.moe_draft_n_expert  &&
             arch  == other.arch  &&
             gtype == other.gtype &&
             cvec  == other.cvec  &&
@@ -717,6 +724,9 @@
     const int64_t n_expert;
     const int64_t n_expert_used;
 
+    // self-speculative decoding: override for n_expert_used in draft mode
+    const int32_t moe_draft_n_expert;
+
     const float freq_base;
     const float freq_scale;
     const float ext_factor;
--- a/src/llama-graph.cpp
+++ b/src/llama-graph.cpp
@@ -829,6 +829,7 @@
     n_embd_v_gqa     (hparams.n_embd_v_gqa()),
     n_expert         (hparams.n_expert),
     n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
+    moe_draft_n_expert(params.moe_draft_n_expert),
     freq_base        (cparams.rope_freq_base),
     freq_scale       (cparams.rope_freq_scale),
     ext_factor       (cparams.yarn_ext_factor),
--- a/src/llama-context.h
+++ b/src/llama-context.h
@@ -105,6 +105,10 @@
     void set_causal_attn(bool value);
     void set_warmup(bool value);
 
+    // self-speculative decoding: set MoE draft expert count
+    //   -1 = normal mode, 0 = shared expert only, 1+ = top-N routed experts
+    void set_moe_draft_mode(int32_t n_expert);
+
     void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);
 
     bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);
@@ -343,6 +347,9 @@
     // env: LLAMA_GRAPH_REUSE_DISABLE
     bool graph_reuse_disable = false;
 
+    // self-speculative decoding: MoE draft expert count (-1 = normal)
+    int32_t moe_draft_n_expert = -1;
+
     // perf
     mutable int64_t t_start_us  = 0;
     mutable int64_t t_load_us   = 0;
--- a/src/llama-context.cpp
+++ b/src/llama-context.cpp
@@ -1013,6 +1013,19 @@
     //sched_need_reserve = true;
 }
 
+void llama_context::set_moe_draft_mode(int32_t n_expert) {
+    LLAMA_LOG_DEBUG("%s: n_expert = %d\n", __func__, n_expert);
+
+    if (moe_draft_n_expert == n_expert) {
+        return;
+    }
+
+    moe_draft_n_expert = n_expert;
+
+    // changing expert count changes graph topology — force scheduler re-reserve
+    sched_need_reserve = true;
+}
+
 bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
     if (!sampler && sampling.samplers.count(seq_id) == 0) {
         return true;
@@ -2107,6 +2120,7 @@
         /*.cross       =*/ &cross,
         /*.samplers    =*/ sampling.samplers,
         /*.n_outputs   =*/ n_outputs,
+        /*.moe_draft_n_expert =*/ moe_draft_n_expert,
         /*.cb          =*/ graph_get_cb(),
         /*.res         =*/ res,
     };
@@ -3128,6 +3142,10 @@
     ctx->set_warmup(warmup);
 }
 
+void llama_set_moe_draft_mode(llama_context * ctx, int32_t n_expert) {
+    ctx->set_moe_draft_mode(n_expert);
+}
+
 void llama_synchronize(llama_context * ctx) {
     ctx->synchronize();
 }
--- a/src/models/deepseek2.cpp
+++ b/src/models/deepseek2.cpp
@@ -207,34 +207,41 @@
                 NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
             cb(cur, "ffn_out", il);
         } else {
-            // MoE branch
-            ggml_tensor * moe_out = build_moe_ffn(cur,
-                model.layers[il].ffn_gate_inp,
-                model.layers[il].ffn_up_exps,
-                model.layers[il].ffn_gate_exps,
-                model.layers[il].ffn_down_exps,
-                model.layers[il].ffn_exp_probs_b,
-                n_expert, n_expert_used,
-                LLM_FFN_SILU, hparams.expert_weights_norm,
-                hparams.expert_weights_scale, hparams.expert_weights_scale,
-                (llama_expert_gating_func_type) hparams.expert_gating_func,
-                il,
-                nullptr,
-                model.layers[il].ffn_gate_up_exps);
-            cb(moe_out, "ffn_moe_out", il);
+            // FFN shared expert (always computed, even in draft mode)
+            ggml_tensor * ffn_shexp =
+                build_ffn(cur,
+                    model.layers[il].ffn_up_shexp, NULL, NULL,
+                    model.layers[il].ffn_gate_shexp, NULL, NULL,
+                    model.layers[il].ffn_down_shexp, NULL, NULL,
+                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
+            cb(ffn_shexp, "ffn_shexp", il);
 
-            // FFN shared expert
-            {
-                ggml_tensor * ffn_shexp =
-                    build_ffn(cur,
-                        model.layers[il].ffn_up_shexp, NULL, NULL,
-                        model.layers[il].ffn_gate_shexp, NULL, NULL,
-                        model.layers[il].ffn_down_shexp, NULL, NULL,
-                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
-                cb(ffn_shexp, "ffn_shexp", il);
+            if (moe_draft_n_expert != 0) {
+                // MoE branch: use full experts (normal) or reduced count (draft)
+                const int64_t n_expert_act = (moe_draft_n_expert > 0)
+                    ? (int64_t) moe_draft_n_expert : n_expert_used;
+
+                ggml_tensor * moe_out = build_moe_ffn(cur,
+                    model.layers[il].ffn_gate_inp,
+                    model.layers[il].ffn_up_exps,
+                    model.layers[il].ffn_gate_exps,
+                    model.layers[il].ffn_down_exps,
+                    model.layers[il].ffn_exp_probs_b,
+                    n_expert, n_expert_act,
+                    LLM_FFN_SILU, hparams.expert_weights_norm,
+                    hparams.expert_weights_scale, hparams.expert_weights_scale,
+                    (llama_expert_gating_func_type) hparams.expert_gating_func,
+                    il,
+                    nullptr,
+                    model.layers[il].ffn_gate_up_exps);
+                cb(moe_out, "ffn_moe_out", il);
 
                 cur = ggml_add(ctx0, moe_out, ffn_shexp);
                 cb(cur, "ffn_out", il);
+            } else {
+                // Self-speculative draft mode: shared expert only, skip routed experts
+                cur = ffn_shexp;
+                cb(cur, "ffn_out", il);
             }
         }
         cur = ggml_add(ctx0, cur, ffn_inp);
--- a/src/models/deepseek.cpp
+++ b/src/models/deepseek.cpp
@@ -91,32 +91,39 @@
                     NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
             cb(cur, "ffn_out", il);
         } else {
-            // MoE branch
-            ggml_tensor * moe_out = build_moe_ffn(cur,
-                model.layers[il].ffn_gate_inp,
-                model.layers[il].ffn_up_exps,
-                model.layers[il].ffn_gate_exps,
-                model.layers[il].ffn_down_exps,
-                nullptr,
-                n_expert, n_expert_used,
-                LLM_FFN_SILU, false,
-                false, hparams.expert_weights_scale,
-                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
-                il);
-            cb(moe_out, "ffn_moe_out", il);
+            // FFN shared expert (always computed, even in draft mode)
+            ggml_tensor * ffn_shexp =
+                build_ffn(cur,
+                    model.layers[il].ffn_up_shexp, NULL, NULL,
+                    model.layers[il].ffn_gate_shexp, NULL, NULL,
+                    model.layers[il].ffn_down_shexp, NULL, NULL,
+                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
+            cb(ffn_shexp, "ffn_shexp", il);
+
+            if (moe_draft_n_expert != 0) {
+                // MoE branch: use full experts (normal) or reduced count (draft)
+                const int64_t n_expert_act = (moe_draft_n_expert > 0)
+                    ? (int64_t) moe_draft_n_expert : n_expert_used;
 
-            // FFN shared expert
-            {
-                ggml_tensor * ffn_shexp =
-                    build_ffn(cur,
-                        model.layers[il].ffn_up_shexp, NULL, NULL,
-                        model.layers[il].ffn_gate_shexp, NULL, NULL,
-                        model.layers[il].ffn_down_shexp, NULL, NULL,
-                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
-                cb(ffn_shexp, "ffn_shexp", il);
+                ggml_tensor * moe_out = build_moe_ffn(cur,
+                    model.layers[il].ffn_gate_inp,
+                    model.layers[il].ffn_up_exps,
+                    model.layers[il].ffn_gate_exps,
+                    model.layers[il].ffn_down_exps,
+                    nullptr,
+                    n_expert, n_expert_act,
+                    LLM_FFN_SILU, false,
+                    false, hparams.expert_weights_scale,
+                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
+                    il);
+                cb(moe_out, "ffn_moe_out", il);
 
                 cur = ggml_add(ctx0, moe_out, ffn_shexp);
                 cb(cur, "ffn_out", il);
+            } else {
+                // Self-speculative draft mode: shared expert only, skip routed experts
+                cur = ffn_shexp;
+                cb(cur, "ffn_out", il);
             }
         }
         cur = ggml_add(ctx0, cur, ffn_inp);
```