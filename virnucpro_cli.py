#!/usr/bin/env python3
"""Command-line interface for VirNucPro viral sequence classifier."""
import argparse
import logging
import os
import sys
import traceback

import virnucpro


def setup_logging(verbose=False):
    """
    Configure logging for CLI.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )


def get_version():
    """Get VirNucPro version from file or env var."""
    version = os.environ.get('VIRNUCPRO_VERSION')
    if version:
        return version
    version_file = '/tmp/virnucpro_version.txt'
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            return f.read().strip()
    return 'unknown'


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='VirNucPro: Classify viral sequences using DNABERT_S and ESM2-3B models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic prediction with 500bp model (BAM input)
  virnucpro_cli.py input.bam output.tsv

  # FASTA input (automatically detected by extension)
  virnucpro_cli.py sequences.fasta output.tsv

  # Use 300bp model with CPU only
  virnucpro_cli.py input.bam output.tsv --expected-length 300 --no-gpu

  # Use multiple GPUs in parallel
  virnucpro_cli.py input.bam output.tsv --gpus 0,1,2,3 --parallel

  # Custom batch sizes for memory-constrained systems
  virnucpro_cli.py input.bam output.tsv --dnabert-batch-size 1024 --esm-batch-size 1024
'''
    )

    parser.add_argument(
        'input_file',
        help='Input file: BAM (.bam) or FASTA (.fasta, .fa, .fna, .ffn, .faa, .frn)'
    )

    parser.add_argument(
        'output_tsv',
        help='Output TSV file with predictions'
    )

    parser.add_argument(
        '--expected-length',
        type=int,
        choices=[300, 500],
        default=500,
        help='Expected sequence length (bp)'
    )

    # GPU/Device options
    gpu_group = parser.add_argument_group('GPU Options')
    gpu_group.add_argument(
        '--use-gpu',
        action='store_true',
        help='Force GPU usage'
    )

    gpu_group.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU usage (disable GPU)'
    )

    gpu_group.add_argument(
        '--gpus',
        type=str,
        default=None,
        help='Comma-separated GPU IDs to use (e.g., "0,1,2"). Overrides CUDA_VISIBLE_DEVICES.'
    )

    gpu_group.add_argument(
        '--parallel',
        action='store_true',
        help='Enable multi-GPU parallel processing for feature extraction'
    )

    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for prediction DataLoader'
    )

    perf_group.add_argument(
        '--dnabert-batch-size',
        type=int,
        default=None,
        help='Token batch size for DNABERT-S processing (default: 2048)'
    )

    perf_group.add_argument(
        '--esm-batch-size',
        type=int,
        default=None,
        help='Token batch size for ESM-2 processing (default: 2048). Reduce if encountering OOM errors.'
    )

    perf_group.add_argument(
        '--threads',
        type=int,
        default=None,
        help='Number of CPU threads for translation and merge (default: auto-detect)'
    )

    perf_group.add_argument(
        '--persistent-models',
        action='store_true',
        help='Keep models loaded in GPU memory between pipeline stages (reduces loading overhead but uses more memory)'
    )

    # Other options
    parser.add_argument(
        '--virnucpro-path',
        help='Path to VirNucPro installation (default: $VIRNUCPRO_PATH)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--version',
        action='version',
        version=get_version()
    )

    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    if args.use_gpu and args.no_gpu:
        logging.error("Cannot specify both --use-gpu and --no-gpu")
        sys.exit(1)

    gpu_mode = True if args.use_gpu else (False if args.no_gpu else None)

    try:
        tool = virnucpro.VirNucPro(virnucpro_path=args.virnucpro_path)
        input_type = tool.detect_input_type(args.input_file)

        classify_args = dict(
            out_report=args.output_tsv,
            expected_length=args.expected_length,
            use_gpu=gpu_mode,
            parallel=args.parallel,
            gpus=args.gpus,
            batch_size=args.batch_size,
            dnabert_batch_size=args.dnabert_batch_size,
            esm_batch_size=args.esm_batch_size,
            threads=args.threads,
            persistent_models=args.persistent_models
        )

        if input_type == 'fasta':
            tool.classify_fasta(args.input_file, **classify_args)
        else:
            tool.classify(args.input_file, **classify_args)

        logging.info("Classification complete: %s", args.output_tsv)
    except Exception as e:
        logging.error("Classification failed: %s", e)
        # Print full traceback to stderr for Cromwell visibility
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
