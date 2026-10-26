"""Mixture-of-experts pieces that two or more model families share.

The top-k router and expert MLP, the sigmoid group-limited gate, the grouped
expert dispatch, and the FP8 and FP4 grouped-GEMM kernels (Triton and Gluon).
"""
