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
    FASTA_EXTENSIONS = {'.fasta', '.fa', '.fna', '.ffn', '.faa', '.frn'}
    BAM_EXTENSIONS = {'.bam'}

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
                 esm_batch_size=None, threads=None, persistent_models=False,
                 resume=False, v1_fallback=False, v1_attention=False):
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
            persistent_models: Keep models loaded in GPU memory between stages.
            resume: Resume from checkpoint if available.
            v1_fallback: Use v1.0 multi-worker architecture for ESM-2.
            v1_attention: Use v1.0-compatible standard attention (exact match, slower).
        """
        with pysam.AlignmentFile(in_bam, 'rb', check_sq=False) as bam:
            is_empty = sum(1 for _ in bam) == 0

        if is_empty:
            log.warning("Input BAM is empty, creating empty output reports")
            with open(out_report, 'wt') as outf:
                outf.write("Sequence_ID\tPrediction\tscore1\tscore2\n")
            # Also create empty consensus file
            out_base, out_ext = os.path.splitext(out_report)
            consensus_out = f"{out_base}_highestscore.csv"
            with open(consensus_out, 'wt') as outf:
                outf.write("Sequence_ID,Prediction,score1,score2\n")
            return

        model_path = self.get_model_path(expected_length)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fasta = os.path.join(tmp_dir, 'input.fasta')

            self._bam_to_fasta(in_bam, tmp_fasta)

            self._run_prediction(
                tmp_fasta, expected_length, model_path, use_gpu=use_gpu,
                parallel=parallel, gpus=gpus, batch_size=batch_size,
                dnabert_batch_size=dnabert_batch_size, esm_batch_size=esm_batch_size,
                threads=threads, output_dir=tmp_dir, persistent_models=persistent_models,
                resume=resume, v1_fallback=v1_fallback, v1_attention=v1_attention
            )

            # New VirNucPro outputs to {output_dir}/input_merged/
            results_dir = os.path.join(tmp_dir, 'input_merged')
            results_file = os.path.join(results_dir, 'prediction_results.txt')
            consensus_file = os.path.join(results_dir, 'prediction_results_highestscore.csv')

            if not os.path.exists(results_file):
                raise RuntimeError(f"VirNucPro did not produce expected output file: {results_file}")

            # Copy main results
            shutil.copy(results_file, out_report)
            log.info("Results saved to %s", out_report)

            # Copy consensus results (derive filename from out_report)
            if os.path.exists(consensus_file):
                out_base, out_ext = os.path.splitext(out_report)
                consensus_out = f"{out_base}_highestscore.csv"
                shutil.copy(consensus_file, consensus_out)
                log.info("Consensus results saved to %s", consensus_out)

    def detect_input_type(self, input_file):
        """
        Detect input file type based on extension.

        Args:
            input_file: Path to input file.

        Returns:
            'bam' or 'fasta'

        Raises:
            ValueError: If file extension is not recognized.
        """
        ext = os.path.splitext(input_file)[1].lower()
        if ext in self.BAM_EXTENSIONS:
            return 'bam'
        elif ext in self.FASTA_EXTENSIONS:
            return 'fasta'
        else:
            raise ValueError(
                f"Unrecognized file extension '{ext}'. "
                f"Supported: BAM ({', '.join(self.BAM_EXTENSIONS)}), "
                f"FASTA ({', '.join(sorted(self.FASTA_EXTENSIONS))})"
            )

    def _is_empty_fasta(self, fasta_path):
        """
        Check if FASTA file is empty (no sequences).

        Args:
            fasta_path: Path to FASTA file.

        Returns:
            True if file has no sequences, False otherwise.
        """
        with open(fasta_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    return False
        return True

    def classify_fasta(self, in_fasta, out_report, expected_length=500, use_gpu=None,
                       parallel=False, gpus=None, batch_size=None, dnabert_batch_size=None,
                       esm_batch_size=None, threads=None, persistent_models=False,
                       resume=False, v1_fallback=False, v1_attention=False):
        """
        Classify sequences directly from FASTA file using VirNucPro.

        Args:
            in_fasta: Input FASTA file with sequences.
            out_report: Output classification report (TSV format).
            expected_length: Expected sequence length (300 or 500, default 500).
            use_gpu: GPU usage control. True=force GPU, False=force CPU, None=auto-detect.
            parallel: Enable multi-GPU parallel processing.
            gpus: Comma-separated GPU IDs to use (e.g., "0,1,2").
            batch_size: Batch size for prediction DataLoader.
            dnabert_batch_size: Token batch size for DNABERT-S processing.
            esm_batch_size: Token batch size for ESM-2 processing.
            threads: Number of CPU threads for translation and merge.
            persistent_models: Keep models loaded in GPU memory between stages.
            resume: Resume from checkpoint if available.
            v1_fallback: Use v1.0 multi-worker architecture for ESM-2.
            v1_attention: Use v1.0-compatible standard attention (exact match, slower).

        Note:
            FASTA sequence IDs must be unique. For paired-end data, add /1 and /2
            suffixes to distinguish read pairs (e.g., >read001/1, >read001/2).
        """
        if self._is_empty_fasta(in_fasta):
            log.warning("Input FASTA is empty, creating empty output reports")
            with open(out_report, 'wt') as outf:
                outf.write("Sequence_ID\tPrediction\tscore1\tscore2\n")
            out_base, out_ext = os.path.splitext(out_report)
            consensus_out = f"{out_base}_highestscore.csv"
            with open(consensus_out, 'wt') as outf:
                outf.write("Sequence_ID,Prediction,score1,score2\n")
            return

        model_path = self.get_model_path(expected_length)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Copy FASTA to temp directory with consistent name for output path derivation
            tmp_fasta = os.path.join(tmp_dir, 'input.fasta')
            shutil.copy(in_fasta, tmp_fasta)

            self._run_prediction(
                tmp_fasta, expected_length, model_path, use_gpu=use_gpu,
                parallel=parallel, gpus=gpus, batch_size=batch_size,
                dnabert_batch_size=dnabert_batch_size, esm_batch_size=esm_batch_size,
                threads=threads, output_dir=tmp_dir, persistent_models=persistent_models,
                resume=resume, v1_fallback=v1_fallback, v1_attention=v1_attention
            )

            # VirNucPro outputs to {output_dir}/input_merged/
            results_dir = os.path.join(tmp_dir, 'input_merged')
            results_file = os.path.join(results_dir, 'prediction_results.txt')
            consensus_file = os.path.join(results_dir, 'prediction_results_highestscore.csv')

            if not os.path.exists(results_file):
                raise RuntimeError(f"VirNucPro did not produce expected output file: {results_file}")

            shutil.copy(results_file, out_report)
            log.info("Results saved to %s", out_report)

            if os.path.exists(consensus_file):
                out_base, out_ext = os.path.splitext(out_report)
                consensus_out = f"{out_base}_highestscore.csv"
                shutil.copy(consensus_file, consensus_out)
                log.info("Consensus results saved to %s", consensus_out)

    def _bam_to_fasta(self, in_bam, out_fasta):
        """
        Convert BAM to FASTA format.

        Paired-end reads get /1 or /2 suffix to ensure unique IDs.
        This matches standard conventions (e.g., samtools fasta output).

        Args:
            in_bam: Input BAM file path.
            out_fasta: Output FASTA file path.
        """
        with pysam.AlignmentFile(in_bam, 'rb', check_sq=False) as bam, open(out_fasta, 'w') as fasta:
            for read in bam:
                name = read.query_name
                if read.is_paired:
                    name += "/1" if read.is_read1 else "/2"
                fasta.write(f">{name}\n{read.query_sequence}\n")

    def _run_prediction(self, fasta_file, expected_length, model_path, use_gpu=None,
                        parallel=False, gpus=None, batch_size=None, dnabert_batch_size=None,
                        esm_batch_size=None, threads=None, output_dir=None,
                        persistent_models=False, resume=False, v1_fallback=False,
                        v1_attention=False):
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
            persistent_models: Keep models loaded in GPU memory between stages.
            resume: Resume from checkpoint if available.
            v1_fallback: Use v1.0 multi-worker architecture for ESM-2.
            v1_attention: Use v1.0-compatible standard attention (exact match, slower).
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

        # Persistent models option
        if persistent_models:
            cmd.append('--persistent-models')

        # v2.0 options
        if resume:
            cmd.append('--resume')
        if v1_fallback:
            cmd.append('--v1-fallback')
        if v1_attention:
            cmd.append('--v1-attention')

        # WHY CUDA_VISIBLE_DEVICES: Standard PyTorch pattern for CPU/GPU control.
        # Setting to "-1" forces CPU mode when GPU unavailable. Cloud VMs may lack GPU.
        env = os.environ.copy()
        if use_gpu is False:
            env['CUDA_VISIBLE_DEVICES'] = '-1'

        log.debug('Running VirNucPro: %s', ' '.join(cmd))

        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # Always print stdout/stderr directly to ensure visibility in Cromwell logs
        if stdout:
            print(stdout, file=sys.stdout)
            log.debug("VirNucPro stdout: %s", stdout)
        if stderr:
            print(stderr, file=sys.stderr)
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
