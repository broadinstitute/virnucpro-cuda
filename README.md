# VirNucPro Standalone Docker Container

VirNucPro is a viral sequence classifier using DNABERT_S and ESM2-3B language models for identifying short viral sequences (300bp or 500bp). This standalone Docker container provides GPU-accelerated classification with **multi-GPU parallel processing support**, automatic CPU fallback, and is designed for cloud pipeline deployment (GCE/dsub/Cromwell/WDL).

**Features:**
- Multi-GPU parallel processing for 150-380x speedup
- Checkpoint-based resume for interrupted runs
- Memory optimization for large datasets
- Backward-compatible BAM input interface

## Architecture

```
User (Cloud Pipeline)
    |
    | docker run virnucpro:latest input.bam output.tsv
    v
+---------------------------+
| Docker Container          |
|                           |
|  +-----------------+      |
|  | CLI Entry Point |      |
|  | (virnucpro_cli.py)|    |
|  +-----------------+      |
|        |                  |
|        v                  |
|  +-----------------+      |
|  | Core Module     |      |
|  | (virnucpro.py)  |      |
|  +-----------------+      |
|        |                  |
|        | BAM input        |
|        v                  |
|  +-----------------+      |
|  | pysam           |      |
|  | BAM → FASTA     |      |
|  +-----------------+      |
|        |                  |
|        | FASTA (with dups)|
|        v                  |
|  +-----------------+      |
|  | Deduplication   |      |
|  | _ensure_unique  |      |
|  +-----------------+      |
|        |                  |
|        | FASTA (unique)   |
|        v                  |
|  +-----------------+      |
|  | VirNucPro CLI   |      |
|  | subprocess call |      |
|  +-----------------+      |
|        |                  |
|        v                  |
|  +------------------+     |
|  | python -m        |     |
|  | virnucpro predict|     |
|  +------------------+     |
|        |                  |
|        | Multi-GPU        |
|        | Parallel         |
|        v                  |
|  +------------------+     |
|  | DNABERT_S +      |     |
|  | ESM2-3B          |     |
|  | Feature Extract  |     |
|  +------------------+     |
|        |                  |
|        | Classification   |
|        v                  |
|  +------------------+     |
|  | Results TSV      |     |
|  +------------------+     |
|        |                  |
+---------------------------+
         |
         | output.tsv
         v
    User receives TSV
```

## Data Flow

```
Input: BAM file (unaligned reads, may have duplicate IDs)
   |
   | pysam converts to FASTA
   v
FASTA (nucleotide sequences, possibly duplicate IDs)
   |
   | _ensure_unique_fasta_ids() adds _N suffix to duplicates
   v
FASTA (unique sequence IDs required by ESM model)
   |
   | Written to temp directory
   v
Subprocess: python -m virnucpro predict input.fasta --model-type 300|500
   |
   | VirNucPro performs (with optional multi-GPU parallel):
   |  - Sequence chunking to expected length
   |  - Six-frame translation (CPU parallel)
   |  - DNABERT_S embedding (nucleotide, GPU parallel)
   |  - ESM2-3B embedding (amino acid, GPU parallel)
   |  - Feature concatenation (CPU parallel)
   |  - MLP classification
   v
Intermediate: input_unique_merged/prediction_results.txt
   |
   | Copy to user-specified output path
   v
Output: TSV file
Sequence_ID    Prediction    score1    score2
read1          virus         0.95      0.05
read1_1        non-virus     0.12      0.88
```

## Usage

### Basic Classification

```bash
# Basic usage (500bp, auto-detect GPU)
docker run -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv

# Specify 300bp model
docker run -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --expected-length 300

# Force CPU mode
docker run -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --no-gpu

# Force GPU mode
docker run --gpus all -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --use-gpu
```

### Multi-GPU Parallel Processing

For large datasets, leverage multi-GPU parallel processing for 150-380x speedup:

```bash
# Use all available GPUs in parallel mode
docker run --gpus all -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --parallel

# Specify specific GPUs
docker run --gpus all -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --gpus 0,1,2,3 --parallel

# Custom batch sizes for memory-constrained systems
docker run --gpus all -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --dnabert-batch-size 1024 --esm-batch-size 512

# Specify CPU threads for translation and merge steps
docker run --gpus all -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --threads 16
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--expected-length` | Expected sequence length: 300 or 500 (default: 500) |
| `--use-gpu` | Force GPU usage |
| `--no-gpu` | Force CPU usage (disable GPU) |
| `--gpus` | Comma-separated GPU IDs (e.g., "0,1,2") |
| `--parallel` | Enable multi-GPU parallel processing |
| `--batch-size` | Batch size for prediction DataLoader |
| `--dnabert-batch-size` | Token batch size for DNABERT-S (default: 2048) |
| `--esm-batch-size` | Token batch size for ESM-2 (default: 2048) |
| `--threads` | CPU threads for translation and merge |
| `--verbose` | Enable debug logging |
| `--virnucpro-path` | Path to VirNucPro installation |

### Input Format

**BAM File** (unaligned reads):
- Unmapped BAM format (SAM flags indicate unmapped status)
- Paired-end reads supported (duplicate read names handled automatically)
- Empty BAM files produce header-only TSV output

### Output Format

**TSV File** (tab-separated values):
- Header: `Sequence_ID\tPrediction\tscore1\tscore2`
- Columns:
  - `Sequence_ID`: Original read ID (with `_N` suffix if deduplicated)
  - `Prediction`: Classification result (`virus` or `non-virus`)
  - `score1`: Confidence score for virus class
  - `score2`: Confidence score for non-virus class

Example output:
```
Sequence_ID    Prediction    score1    score2
read1          virus         0.95      0.05
read1_1        non-virus     0.12      0.88
read2          virus         0.87      0.13
```

## System Requirements

- **RAM**: Minimum 8GB
- **GPU**: Optional; CUDA-compatible GPU (V100/T4/A100) for accelerated inference
- **Storage**: ~4GB for Docker image
- **CPU Fallback**: Automatic via `CUDA_VISIBLE_DEVICES="-1"` when GPU unavailable

## Design Decisions

### Separate CLI and Core Module

- **`virnucpro_cli.py`**: Thin entry point handling argparse, logging setup, exception handling
- **`virnucpro.py`**: Reusable VirNucPro class with classify() method
- **Benefit**: Testing core logic without CLI overhead; reusable VirNucPro class in other scripts; clear responsibility boundary

### VirNucPro as Subprocess

VirNucPro uses `python -m virnucpro predict` for command-line invocation. Subprocess execution provides:
- Isolation preventing PyTorch memory leaks in wrapper process
- Consistency with viral-classify pattern for tool integration
- Negligible overhead (~50-100ms process spawn) for minute-scale PyTorch inference
- Clean separation between BAM handling (wrapper) and ML inference (VirNucPro)

### Unique FASTA ID Enforcement

ESM model (from fair-esm library) crashes on duplicate sequence IDs. BAM files inherently have duplicate read names for paired-end reads. The `_ensure_unique_fasta_ids()` function:
- Appends `_N` suffix to duplicates (read1, read1_1, read1_2...)
- Prevents ESM model crashes
- Enables traceability via prefix matching
- Logs warning with deduplication count

### Multi-Stage Docker Build

- **Stage 1 (builder)**: Install build dependencies (gcc, git), clone VirNucPro, install Python packages
- **Stage 2 (runtime)**: Copy only /opt/VirNucPro and Python packages, exclude build tools
- **Result**: Image size reduced from ~5GB to ~3.5GB; faster cloud VM startup; lower registry storage costs

### Environment Variable Configuration

- **`VIRNUCPRO_VERSION`**: Git commit SHA for version tracking
- **`VIRNUCPRO_PATH`**: Installation directory (/opt/VirNucPro)
- **`CUDA_VISIBLE_DEVICES`**: GPU device selection, "-1" for CPU mode
- Pattern from beast2-beagle-cuda allows version matrix builds

## Invariants

### FASTA ID Uniqueness

After `_ensure_unique_fasta_ids()`, all FASTA sequence IDs must be unique within file. ESM model will crash if duplicate IDs present. Deduplication must preserve original ID as prefix for traceability.

### Sequence Length Matching

`expected_length` parameter (300 or 500) must match model file:
- 300bp sequences require 300_model.pth
- 500bp sequences require 500_model.pth
- Mismatch produces invalid predictions (VirNucPro silently accepts but results meaningless)

### Temp Directory Cleanup

VirNucPro creates subdirectories: `{prefix}_nucleotide/`, `{prefix}_protein/`, `{prefix}_merged/`. All temporary files must be cleaned up after `classify()` completes to prevent disk space exhaustion in long-running cloud jobs.

### Empty Input Handling

Empty BAM must produce TSV with header line only:
- Header format: `Sequence_ID\tPrediction\tscore1\tscore2\n`
- Zero data rows acceptable for downstream tools
- Prevents pipeline failures on empty input splits

### Subprocess Error Detection

Must check both process return code AND stderr for "Traceback". Python exceptions don't always set non-zero exit codes. Pattern from viral-classify classify/kb.py.

## Tradeoffs

### Multi-Stage Build (Size vs Complexity)

- **Cost**: More complex Dockerfile, longer build time (two stages)
- **Benefit**: Image size reduced ~30% (5GB → 3.5GB), faster cloud VM startup
- **Decision**: Cloud deployment benefits outweigh build complexity

### Unique FASTA ID Deduplication (Silent vs Error)

- **Cost**: Changes sequence IDs from input, users must track _N suffixes
- **Benefit**: Prevents ESM model crashes, user can't control paired-end BAM format
- **Decision**: Robustness over strict ID preservation, logging provides traceability

### Python Wrapper vs Bash (Maintainability vs Simplicity)

- **Cost**: Python adds ~200 lines vs ~50 lines bash, requires Python testing
- **Benefit**: Better string manipulation, code reuse from viral-classify, easier testing
- **Decision**: Maintainability prioritized, Python not significantly more complex

### Bundled Models vs Download-on-Demand (Image Size vs Offline)

- **Cost**: 14MB in image, models frozen at build time
- **Benefit**: No internet dependency, immediate execution, reproducible
- **Decision**: 14MB negligible relative to 2.5GB PyTorch, offline preferred

### Include Samtools (Image Size vs UX)

- **Cost**: ~50MB added to image
- **Benefit**: Users don't need separate BAM→FASTA conversion step
- **Decision**: UX improvement worth minimal size increase

## Development

### Building Locally

```bash
# Clone repository
git clone <repository-url>
cd virnucpro-cuda

# Build Docker image
docker build -t virnucpro:latest .

# Verify build
docker run virnucpro:latest python --version
docker run virnucpro:latest ls /opt/VirNucPro
```

### Running Tests

**Unit Tests** (requires pytest):
```bash
# Install dependencies
pip install pytest pytest-mock pysam

# Run unit tests
pytest tests/test_virnucpro.py tests/test_cli.py
```

**Integration Tests** (requires Docker):
```bash
# Build test image
docker build -t virnucpro:test .

# Run integration tests
pytest tests/integration/

# Or use helper script
./tests/integration/run_integration.sh
```

## Deployment

### Cloud Pipelines (GCE/dsub)

```bash
# Example dsub command
dsub \
  --provider google-v2 \
  --project <project-id> \
  --regions us-central1 \
  --logging gs://<bucket>/logs \
  --image quay.io/broadinstitute/virnucpro:latest \
  --input INPUT_BAM=gs://<bucket>/input.bam \
  --output OUTPUT_TSV=gs://<bucket>/output.tsv \
  --command '/opt/virnucpro_cli.py ${INPUT_BAM} ${OUTPUT_TSV} --expected-length 500 --no-gpu'
```

### Version Management

Docker images tagged with git commit SHA and `latest`:
- **Latest**: `quay.io/broadinstitute/virnucpro:latest` (main branch)
- **Specific version**: `quay.io/broadinstitute/virnucpro:<commit-sha>`
- **VIRNUCPRO_VERSION**: Environment variable contains git commit SHA at build time

To pin to specific version:
```bash
docker pull quay.io/broadinstitute/virnucpro:<commit-sha>
docker run quay.io/broadinstitute/virnucpro:<commit-sha> ...
```

### GPU Configuration

**GPU-enabled VMs**:
```bash
# Auto-detect GPU (default)
docker run --gpus all -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv

# Force GPU usage (fail if GPU unavailable)
docker run --gpus all -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --use-gpu
```

**CPU-only VMs**:
```bash
# Force CPU mode (no GPU required)
docker run -v $(pwd):/data virnucpro:latest /opt/virnucpro_cli.py /data/input.bam /data/output.tsv --no-gpu
```

## Known Issues

### CI/CD Docker Image Tag Mismatch

**Impact**: Integration tests may fail in GitHub Actions CI but pass locally.

**Details**: The CI workflow sets `VIRNUCPRO_DOCKER_IMAGE=virnucpro-cuda:test` (`.github/workflows/test.yml:63`), but the integration test fixture builds the image as `virnucpro:test` (`tests/integration/test_integration.py:23`). This mismatch causes the tests to build a new image rather than using the CI-provided one.

**Workaround**: Tests will still pass by building the image themselves. This only affects CI pipeline efficiency, not functionality.

**Status**: Accepted as low-priority technical debt (QR Option B). Does not affect production usage.

## License

See LICENSE file for licensing information.

## References

- VirNucPro (Broad refactored): https://github.com/broadinstitute/virnucpro
- VirNucPro (original): https://github.com/Li-Jing-1997/VirNucPro
- DNABERT_S: Language model for nucleotide sequences
- ESM2-3B: Protein language model from fair-esm
