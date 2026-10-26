"""DeepSeek-V4 (`DeepseekV4ForCausalLM`).

The model, its decoder layer, the hybrid attention (HCA: Heavily Compressed
Attention, CSA: Compressed Sparse Attention, and sliding-window attention),
hyper-connections, the hash-routed MoE with FP4 experts, and the attention
primitives that HCA and CSA share (compressor, attention sink, grouped output
projection, Lightning Indexer). V4 paper section 2.3, formulas 20-27.
"""
