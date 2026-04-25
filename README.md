# my_gpt

learning playground for LLM internals

## roadmap

### inference efficiency
- [x] kv cache
- quantization
  - [x] absmax (tensor-wise)
  - [x] zero-point
  - [ ] LLM.int8
  - [ ] 4-bit (NF4 / GPTQ)
  - [ ] quantization-aware training
  - [ ] TurboQuant

### architecture
- [ ] mixture of experts (2017)
- [ ] RMSNorm (2019)
- [ ] multi-query attention (2019)
- [ ] RoPE (2021)
- [ ] grouped-query attention (2023)

### mechanistic interpretability

### tooling
- [ ] benchmark harness (latency / throughput / memory)
- [ ] perplexity eval on validation set
- [ ] activation profiler