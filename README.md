<img width="2456" height="930" alt="image" src="https://github.com/user-attachments/assets/737b0ff8-1f55-4456-bc32-89a532cbd716" />


# NVMAI

## Ornith 1.5

> [!IMPORTANT]
> NVMAI now supports text-only **Ornith-1.5-35B-A3B** installation and inference
> in 4-bit and 8-bit. It reuses the verified Qwen3.5-MoE runtime and keeps routed
> experts SSD-streamed with the same bounded-memory design. Ornith's native MTP
> draft is available as an optional experimental sidecar; current M3 benchmarks
> do not show a speed benefit. Vision is not included, and Qwen 3.6 remains
> supported. **Ornith 1.5 4-bit is now the default installer, app, launcher,
> benchmark, and real-inference test baseline.**

[![Published Ornith 1.5 benchmark overview](assets/stats.png)](https://ornith.ai/ornith_1_5.html)

Ornith is a 35B mixture-of-experts model with approximately 3B active
parameters per token. The chart above is publisher-supplied, not an NVMAI
measurement. See the [Ornith 1.5 announcement](https://ornith.ai/ornith_1_5.html)
and [model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

## Benchmarks

Fastest single-run decode results from the verified Ornith 1.5 benchmark set on
a base M3 MacBook Pro with 24 GB:

| Quantization | Best decode | Prompt |
| --- | ---: | --- |
| 4-bit | **7.34 tok/s** | Short |
| 8-bit | **4.38 tok/s** | Medium |

[Full benchmark results](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)

## Core Links

- [Getting started](https://github.com/Pummelchen/NVMAI/wiki/Getting-Started)
- [Features](https://github.com/Pummelchen/NVMAI/wiki/Features)
- [Local server and launchers](https://github.com/Pummelchen/NVMAI/wiki/OpenAI-Compatible-Server)
- [Runtime controls](https://github.com/Pummelchen/NVMAI/wiki/Runtime-Controls)
- [Benchmarks](https://github.com/Pummelchen/NVMAI/wiki/Benchmarks)
- [Changelog](https://github.com/Pummelchen/NVMAI/wiki/Changelog)

## Credits

NVMAI is a focused fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare), which
provides the bounded-memory runtime, installer, CLI, Mac app, and local server.
The Qwen 3.6 integration was created by
[NeelM0906](https://github.com/NeelM0906) in
[upstream PR #29](https://github.com/drumih/turbo-fieldfare/pull/29). Concise
mode is derived from the
[Nail-Qwen3.6-35B-A3B](https://huggingface.co/peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-MLX)
chat template by [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll).
