"""VirNucPro: Viral sequence classifier using DNABERT_S and ESM2-3B models."""
import logging
import os
import shutil
import subprocess
import sys
import tempfile

import pysam

log = logging.getLogger(__name__)


class VirNucPro:
    """
    VirNucPro classifier using DNABERT_S and ESM2-3B models for viral sequence identification.

    VirNucPro performs six-frame translation and uses large language models to identify
    short viral sequences (300bp or 500bp).
    """

    SUPPORTED_LENGTHS = [300, 500]

    def __init__(self, virnucpro_path=None):
        """
        Initialize VirNucPro wrapper.

        Args:
            virnucpro_path: Path to VirNucPro installation directory.
                           Defaults to $VIRNUCPRO_PATH environment variable or /opt/VirNucPro.
        """
        if virnucpro_path is None:
            virnucpro_path = os.environ.get('VIRNUCPRO_PATH', '/opt/VirNucPro')
        self.virnucpro_path = virnucpro_path
        log.debug('VirNucPro path: %s', self.virnucpro_path)

    def get_model_path(self, expected_length):
        """
        Get path to model file for specified sequence length.

        Args:
            expected_length: Expected sequence length (300 or 500).

        Returns:
            Path to model file.

        Raises:
            ValueError: If expected_length is not in SUPPORTED_LENGTHS.
            FileNotFoundError: If model file does not exist.
        """
        if expected_length not in self.SUPPORTED_LENGTHS:
            raise ValueError(f"Expected length must be one of {self.SUPPORTED_LENGTHS}")

        model_file = f"{expected_length}_model.pth"
        model_path = os.path.join(self.virnucpro_path, model_file)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        return model_path

    def classify(self, in_bam, out_report, expected_length=500, use_gpu=None,
                 parallel=False, gpus=None, batch_size=None, dnabert_batch_size=None,
                 esm_batch_size=None, threads=None):
        """
        Classify reads from BAM file using VirNucPro.

        Args:
            in_bam: Input unaligned reads in BAM format.
            out_report: Output classification report (TSV format).
            expected_length: Expected sequence length (300 or 500, default 500).
            use_gpu: GPU usage control. True=force GPU, False=force CPU, None=auto-detect.
            parallel: Enable multi-GPU parallel processing.
            gpus: Comma-separated GPU IDs to use (e.g., "0,1,2").
            batch_size: Batch size for prediction DataLoader.
            dnabert_batch_size: Token batch size for DNABERT-S processing.
            esm_batch_size: Token batch size for ESM-2 processing.
            threads: Number of CPU threads for translation and merge.
        """
        with pysam.AlignmentFile(in_bam, 'rb', check_sq=False) as bam:
            is_empty = sum(1 for _ in bam) == 0

        if is_empty:
            log.warning("Input BAM is empty, creating empty output report")
            with open(out_report, 'wt') as outf:
                outf.write("Sequence_ID\tPrediction\tscore1\tscore2\n")
            return

        model_path = self.get_model_path(expected_length)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fasta = os.path.join(tmp_dir, 'input.fasta')
            tmp_fasta_unique = os.path.join(tmp_dir, 'input_unique.fasta')

            self._bam_to_fasta(in_bam, tmp_fasta)
            self._ensure_unique_fasta_ids(tmp_fasta, tmp_fasta_unique)

            self._run_prediction(
                tmp_fasta_unique, expected_length, model_path, use_gpu=use_gpu,
                parallel=parallel, gpus=gpus, batch_size=batch_size,
                dnabert_batch_size=dnabert_batch_size, esm_batch_size=esm_batch_size,
                threads=threads, output_dir=tmp_dir
            )

            # New VirNucPro outputs to {output_dir}/input_unique_merged/prediction_results.txt
            results_dir = os.path.join(tmp_dir, 'input_unique_merged')
            results_file = os.path.join(results_dir, 'prediction_results.txt')

            if not os.path.exists(results_file):
                raise RuntimeError(f"VirNucPro did not produce expected output file: {results_file}")

            shutil.copy(results_file, out_report)
            log.info("Results saved to %s", out_report)

    def _bam_to_fasta(self, in_bam, out_fasta):
        """
        Convert BAM to FASTA format.

        Args:
            in_bam: Input BAM file path.
            out_fasta: Output FASTA file path.
        """
        with pysam.AlignmentFile(in_bam, 'rb', check_sq=False) as bam, open(out_fasta, 'w') as fasta:
            for read in bam:
                fasta.write(f">{read.query_name}\n")
                fasta.write(f"{read.query_sequence}\n")

    def _ensure_unique_fasta_ids(self, input_fasta, output_fasta):
        """
        Ensure all FASTA sequence IDs are unique by adding numeric suffixes.

        WHY deduplication: ESM model (from fair-esm library) crashes with "KeyError" when
        duplicate sequence IDs are present. BAM files inherently have duplicate read names
        for paired-end reads (read1/1 and read1/2 both named "read1"). Silent deduplication
        with _N suffix prevents crashes while preserving traceability via prefix matching.

        Args:
            input_fasta: Input FASTA file path (may have duplicate IDs).
            output_fasta: Output FASTA file path (will have unique IDs).
        """
        seen_ids = {}  # Track count of each original ID
        all_ids = set()  # Track ALL IDs (original + generated) to prevent collisions
        total_dups = 0

        with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
            for line in infile:
                if line.startswith('>'):
                    seq_id = line[1:].split()[0]

                    if seq_id in seen_ids:
                        # Duplicate found - generate unique suffix
                        seen_ids[seq_id] += 1
                        total_dups += 1

                        # Keep incrementing suffix until we find an unused ID
                        suffix_num = seen_ids[seq_id]
                        unique_id = f"{seq_id}_{suffix_num}"
                        while unique_id in all_ids:
                            suffix_num += 1
                            unique_id = f"{seq_id}_{suffix_num}"
                        seen_ids[seq_id] = suffix_num
                    else:
                        seen_ids[seq_id] = 0
                        unique_id = seq_id

                    all_ids.add(unique_id)
                    rest_of_header = line[1+len(seq_id):]
                    outfile.write(f">{unique_id}{rest_of_header}")
                else:
                    outfile.write(line)

        if total_dups > 0:
            log.warning("Deduplicated %d duplicate FASTA IDs", total_dups)

    def _run_prediction(self, fasta_file, expected_length, model_path, use_gpu=None,
                        parallel=False, gpus=None, batch_size=None, dnabert_batch_size=None,
                        esm_batch_size=None, threads=None, output_dir=None):
        """
        Run VirNucPro prediction using the refactored CLI.

        WHY subprocess: Subprocess isolation prevents PyTorch memory leaks in long-running
        wrapper process. The new VirNucPro uses `python -m virnucpro predict` as entry point.

        Args:
            fasta_file: Input FASTA file.
            expected_length: Expected sequence length.
            model_path: Path to model file.
            use_gpu: GPU usage control. True=force GPU, False=force CPU, None=auto-detect.
            parallel: Enable multi-GPU parallel processing.
            gpus: Comma-separated GPU IDs to use.
            batch_size: Batch size for prediction DataLoader.
            dnabert_batch_size: Token batch size for DNABERT-S processing.
            esm_batch_size: Token batch size for ESM-2 processing.
            threads: Number of CPU threads for translation and merge.
            output_dir: Output directory for results.
        """
        # Build command for new VirNucPro CLI
        cmd = [
            sys.executable, '-m', 'virnucpro', 'predict',
            fasta_file,
            '--model-type', str(expected_length),
            '--model-path', model_path,
            '--force',  # Overwrite output directory if exists
            '--no-progress',  # Disable progress bars for subprocess
        ]

        # Add output directory if specified
        if output_dir:
            cmd.extend(['--output-dir', output_dir])

        # Device/GPU options
        if use_gpu is False:
            cmd.extend(['--device', 'cpu'])
        elif gpus:
            cmd.extend(['--gpus', gpus])

        # Parallel processing
        if parallel:
            cmd.append('--parallel')

        # Batch size options
        if batch_size:
            cmd.extend(['--batch-size', str(batch_size)])
        if dnabert_batch_size:
            cmd.extend(['--dnabert-batch-size', str(dnabert_batch_size)])
        if esm_batch_size:
            cmd.extend(['--esm-batch-size', str(esm_batch_size)])

        # Thread options
        if threads:
            cmd.extend(['--threads', str(threads)])

        # WHY CUDA_VISIBLE_DEVICES: Standard PyTorch pattern for CPU/GPU control.
        # Setting to "-1" forces CPU mode when GPU unavailable. Cloud VMs may lack GPU.
        env = os.environ.copy()
        if use_gpu is False:
            env['CUDA_VISIBLE_DEVICES'] = '-1'

        log.debug('Running VirNucPro: %s', ' '.join(cmd))

        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if stdout:
            log.debug("VirNucPro stdout: %s", stdout)
        if stderr:
            log.debug("VirNucPro stderr: %s", stderr)

        # WHY check Traceback in stderr: Python exceptions don't always set non-zero exit codes.
        # Pattern from viral-classify classify/kb.py ensures we catch all failures.
        has_error = stderr and 'Traceback' in stderr

        if process.returncode != 0 or has_error:
            if stderr:
                log.error("VirNucPro error output: %s", stderr)
                raise RuntimeError(f"VirNucPro failed: {stderr}")
            else:
                raise RuntimeError(f"VirNucPro failed with exit code {process.returncode}")
