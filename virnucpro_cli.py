#!/usr/bin/env python3
"""Command-line interface for VirNucPro viral sequence classifier."""
import argparse
import logging
import os
import sys

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
        description='VirNucPro: Classify viral sequences using DNABERT_S and ESM2-3B models.'
    )

    parser.add_argument(
        'input_bam',
        help='Input BAM file (unaligned reads)'
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

    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Force GPU usage'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU usage (disable GPU)'
    )

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
        tool.classify(
            args.input_bam,
            args.output_tsv,
            expected_length=args.expected_length,
            use_gpu=gpu_mode
        )
        logging.info("Classification complete: %s", args.output_tsv)
    except Exception as e:
        logging.error("Classification failed: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
