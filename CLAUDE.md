# CLAUDE.md

Navigation index for VirNucPro standalone Docker container.

| File/Directory | WHAT | WHEN |
|----------------|------|------|
| virnucpro.py | Core classification module with BAM→FASTA→VirNucPro→TSV pipeline, FASTA ID deduplication, subprocess wrapper for prediction.py | When classifying viral sequences in standalone container, implementing core logic |
| virnucpro_cli.py | CLI entry point with argparse for input/output paths, expected length, GPU flags | When invoking VirNucPro from command line, parsing user arguments |
| Dockerfile | Multi-stage build with CUDA 11.8 runtime base, VirNucPro cloning, Python package installation, samtools integration | When building container image, understanding build process |
| tests/test_virnucpro.py | Unit tests for VirNucPro class: model path validation, FASTA ID deduplication, empty BAM handling, GPU control, subprocess errors | When verifying core module functionality with mocks |
| tests/test_cli.py | Unit tests for CLI: argument parsing, help/version display, GPU flag conflicts, exception handling | When verifying command-line interface behavior |
| tests/integration/ | Integration tests with real PyTorch models in Docker: 300bp/500bp classification, empty BAM, CPU mode | When verifying end-to-end functionality with actual dependencies |
| README.md | Architecture diagrams, data flow, usage examples, system requirements, invariants, tradeoffs, deployment guide | When understanding project structure, deploying to cloud pipelines, troubleshooting |
| requirements.txt | Python dependencies for wrapper (pysam for BAM reading) | When installing wrapper dependencies, separate from VirNucPro's requirements |
| .dockerignore | Build exclusions (.git, __pycache__, tests) for smaller context | When optimizing Docker build |
