"""Integration tests for VirNucPro using Docker."""
import os
import shutil
import subprocess
import tempfile

import pytest


@pytest.fixture(scope='session')
def check_docker_available():
    """Check if docker command is available, skip tests if not."""
    if shutil.which('docker') is None:
        pytest.skip('Docker not available')


@pytest.fixture(scope='session')
def build_docker_image(check_docker_available):
    """Build Docker image for testing."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    result = subprocess.run(
        ['docker', 'build', '-t', 'virnucpro:test', '.'],
        cwd=project_root,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        pytest.fail(f'Docker build failed: {result.stderr}')

    return 'virnucpro:test'


@pytest.fixture(scope='session')
def test_data_dir():
    """Return path to test data directory."""
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture(scope='session')
def test_bam_path(test_data_dir):
    """Return path to test BAM file, creating if needed."""
    bam_path = os.path.join(test_data_dir, 'small.bam')

    if not os.path.exists(bam_path):
        create_script = os.path.join(os.path.dirname(__file__), 'create_test_bam.py')
        result = subprocess.run(
            ['python3', create_script],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            pytest.fail(f'Failed to create test BAM: {result.stderr}')

    return bam_path


@pytest.fixture(scope='session')
def empty_bam_path(test_data_dir):
    """Return path to empty BAM file, creating if needed."""
    bam_path = os.path.join(test_data_dir, 'empty.bam')

    if not os.path.exists(bam_path):
        create_script = os.path.join(os.path.dirname(__file__), 'create_test_bam.py')
        result = subprocess.run(
            ['python3', create_script],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            pytest.fail(f'Failed to create empty BAM: {result.stderr}')

    return bam_path


@pytest.mark.parametrize("expected_length", [300, 500])
def test_integration_classification(build_docker_image, test_data_dir, test_bam_path, expected_length):
    """Run classification with specified model length, verify TSV output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_tsv = os.path.join(tmpdir, 'output.tsv')

        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{test_data_dir}:/test_data:ro',
            '-v', f'{tmpdir}:/output',
            build_docker_image,
            '/opt/virnucpro_cli.py',
            '/test_data/small.bam',
            '/output/output.tsv',
            '--expected-length', str(expected_length),
            '--no-gpu'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f'Docker command failed: {result.stderr}'
        assert os.path.exists(output_tsv), 'Output TSV not created'

        with open(output_tsv, 'r') as f:
            lines = f.readlines()

        assert len(lines) > 0, 'Output TSV is empty'
        assert lines[0].strip() == 'Sequence_ID\tPrediction\tscore1\tscore2', \
            f'Unexpected header: {lines[0]}'

        if len(lines) > 1:
            data_line = lines[1].strip()
            columns = data_line.split('\t')
            assert len(columns) == 4, f'Expected 4 columns, got {len(columns)}'


def test_integration_empty_bam(build_docker_image, test_data_dir, empty_bam_path):
    """Run with empty BAM, verify header-only output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_tsv = os.path.join(tmpdir, 'output.tsv')

        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{test_data_dir}:/test_data:ro',
            '-v', f'{tmpdir}:/output',
            build_docker_image,
            '/opt/virnucpro_cli.py',
            '/test_data/empty.bam',
            '/output/output.tsv',
            '--expected-length', '500',
            '--no-gpu'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f'Docker command failed: {result.stderr}'
        assert os.path.exists(output_tsv), 'Output TSV not created'

        with open(output_tsv, 'r') as f:
            content = f.read()

        assert content == 'Sequence_ID\tPrediction\tscore1\tscore2\n', \
            f'Expected header-only output, got: {content}'


def test_integration_gpu_disabled(build_docker_image, test_data_dir, test_bam_path):
    """Verify CPU mode with CUDA_VISIBLE_DEVICES=-1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_tsv = os.path.join(tmpdir, 'output.tsv')

        cmd = [
            'docker', 'run', '--rm',
            '-e', 'CUDA_VISIBLE_DEVICES=-1',
            '-v', f'{test_data_dir}:/test_data:ro',
            '-v', f'{tmpdir}:/output',
            build_docker_image,
            '/opt/virnucpro_cli.py',
            '/test_data/small.bam',
            '/output/output.tsv',
            '--expected-length', '500'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f'Docker command failed: {result.stderr}'
        assert os.path.exists(output_tsv), 'Output TSV not created'

        with open(output_tsv, 'r') as f:
            lines = f.readlines()

        assert len(lines) > 0, 'Output TSV is empty'
        assert lines[0].strip() == 'Sequence_ID\tPrediction\tscore1\tscore2', \
            f'Unexpected header: {lines[0]}'
