# DLTC and Low-Bit Delay Exploration

Repo is an exploration/combination of two papers:

- Sun et al., "Delay Learning based on Temporal Coding in Spiking Neural Networks," *Neural Networks*, 180, 106678, 2024. DOI: [10.1016/j.neunet.2024.106678](https://doi.org/10.1016/j.neunet.2024.106678). This is the practical reference point for learned delays and temporal coding in SNNs.

- Sun et al., "Exploiting heterogeneous delays for efficient computation in low-bit neural networks," arXiv:2510.27434, 2025. arXiv: [2510.27434](https://arxiv.org/abs/2510.27434), DOI: [10.48550/arXiv.2510.27434](https://doi.org/10.48550/arXiv.2510.27434). This supplies the low-bit, memory-efficiency, and hardware-relevant motivation.

## Artifacts

- `scripts/toy_delay_quantization_demo.py`: independent toy experiment for accuracy-memory trade-off analysis under weight and delay quantisation.
- `results/`: generated CSV and plot outputs from the toy demo.

## Running the Toy Demo

From the repository root:

```powershell
python .\snn-delay-quantisation\scripts\toy_delay_quantization_demo.py
```

The script trains a small synthetic temporal-pattern classifier with learnable delays, then evaluates post-training weight and delay quantisation.

## Licensing Note

The DLTC paper points to public code at `https://github.com/sunpengfei1122/DLTC`. Treat that code as a reference for private study unless a clear license is confirmed. Do not publish copied or modified DLTC source code without permission or a license that permits reuse.
