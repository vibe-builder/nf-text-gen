A novice attempt at making a neural field text generator using Fourier features + SIRENs inspired by QTF-Gen (https://github.com/kyegomez/QTF-Gen) 

## Key Components:
1. **FourierEmbed**: Learnable freq embeddings for coords (sin/cos proj)
2. **SIREN**: Enhanced sinusoidal network w/ better init & per-layer ω
3. **NFTokenizer**: Dynamic grad checkpointing based on GPU mem usage
4. **QFTLayer**: Quantum-inspired conv layer (depthwise sep option)
5. **NFGen**: Main model - tokenizes coords → processes → pools → logits

## Optimizations:
- `torch.compile()` fallback via `maybe_compile()`
- AMP w/ `GradScaler` and memory-aware checkpointing
- Depthwise separable convs for QFT layers
- Direct device tensor creation (`gen_coords`)
- Dynamic batching w/ padding masks

## Training Tricks:
- Selective mixed precision (monitors grad norms)
- Dropout ablation study util
- Sinusoidal init variants
- Attention/mean/max pooling options

## Dev Stuff:
- Configurable logging
- Chrome trace profiling
- Unit tests w/ checkpoint verification
- GPU mem monitoring

## Usage:
```python
model = NFGen(vocab_size=5000).cuda()
trainer = NFTrainer(model)
coords = gen_coords(batch=32, seq_len=128)
logits = model(coords)  # [32, 5000]
