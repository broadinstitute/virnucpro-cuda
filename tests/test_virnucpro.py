"""Unit tests for VirNucPro."""
import os

import pysam
import pytest

import virnucpro


@pytest.fixture
def virnucpro_path(tmpdir):
    """Create mock VirNucPro installation directory with required files."""
    virnucpro_dir = tmpdir.mkdir('virnucpro')

    # Create mock model files (required for get_model_path validation)
    model_300 = virnucpro_dir.join('300_model.pth')
    model_300.write('mock model data for 300bp')

    model_500 = virnucpro_dir.join('500_model.pth')
    model_500.write('mock model data for 500bp')

    return str(virnucpro_dir)


@pytest.fixture
def virnucpro_tool(virnucpro_path):
    """Return VirNucPro tool instance with mock installation path."""
    return virnucpro.VirNucPro(virnucpro_path=virnucpro_path)


@pytest.fixture
def empty_bam(tmpdir):
    """Create an empty BAM file."""
    bam_path = str(tmpdir.join('empty.bam'))
    with pysam.AlignmentFile(bam_path, 'wb', header={'HD': {'VN': '1.0'}}) as bam:
        pass
    return bam_path


@pytest.fixture
def test_bam(tmpdir):
    """Create a test BAM file with some reads."""
    bam_path = str(tmpdir.join('test.bam'))
    header = {'HD': {'VN': '1.0'}}
    with pysam.AlignmentFile(bam_path, 'wb', header=header) as bam:
        for i in range(3):
            read = pysam.AlignedSegment()
            read.query_name = f'read{i}'
            read.query_sequence = 'ATCG' * 10
            read.flag = 4
            read.reference_id = -1
            read.reference_start = -1
            read.mapping_quality = 0
            read.cigar = None
            read.next_reference_id = -1
            read.next_reference_start = -1
            read.template_length = 0
            read.query_qualities = pysam.qualitystring_to_array('I' * 40)
            bam.write(read)
    return bam_path


def test_get_model_path_valid(virnucpro_tool):
    """Test get_model_path() returns correct paths for valid lengths."""
    model_path_300 = virnucpro_tool.get_model_path(300)
    assert model_path_300.endswith('300_model.pth')
    assert os.path.exists(model_path_300)

    model_path_500 = virnucpro_tool.get_model_path(500)
    assert model_path_500.endswith('500_model.pth')
    assert os.path.exists(model_path_500)


def test_get_model_path_invalid(virnucpro_tool):
    """Test get_model_path() raises ValueError for unsupported lengths."""
    with pytest.raises(ValueError) as exc_info:
        virnucpro_tool.get_model_path(400)
    assert 'Expected length must be one of [300, 500]' in str(exc_info.value)


def test_get_model_path_missing(tmpdir):
    """Test get_model_path() raises FileNotFoundError when model file missing."""
    empty_dir = tmpdir.mkdir('empty_virnucpro')

    tool = virnucpro.VirNucPro(virnucpro_path=str(empty_dir))

    with pytest.raises(FileNotFoundError) as exc_info:
        tool.get_model_path(300)
    assert 'Model file not found' in str(exc_info.value)


def test_classify_empty_bam(virnucpro_tool, empty_bam, tmpdir):
    """Test classify() creates header-only output when BAM is empty."""
    out_report = str(tmpdir.join('output.txt'))
    virnucpro_tool.classify(empty_bam, out_report, expected_length=500)

    assert os.path.exists(out_report)
    with open(out_report, 'r') as f:
        content = f.read()
        assert content == "Sequence_ID\tPrediction\tscore1\tscore2\n"


def test_ensure_unique_fasta_ids(virnucpro_tool, tmpdir):
    """Test _ensure_unique_fasta_ids() makes duplicate IDs unique."""
    input_fasta = tmpdir.join('input.fasta')
    input_fasta.write(
        '>read1\n'
        'ATCG\n'
        '>read2\n'
        'GCTA\n'
        '>read1\n'
        'TTTT\n'
        '>read1\n'
        'AAAA\n'
        '>read3 extra info here\n'
        'GGGG\n'
    )

    output_fasta = str(tmpdir.join('output.fasta'))
    virnucpro_tool._ensure_unique_fasta_ids(str(input_fasta), output_fasta)

    with open(output_fasta, 'r') as f:
        content = f.read()

    expected = (
        '>read1\n'
        'ATCG\n'
        '>read2\n'
        'GCTA\n'
        '>read1_1\n'
        'TTTT\n'
        '>read1_2\n'
        'AAAA\n'
        '>read3 extra info here\n'
        'GGGG\n'
    )

    assert content == expected


def test_ensure_unique_fasta_ids_collision_prevention(virnucpro_tool, tmpdir):
    """Test collision prevention when suffixed IDs already exist in input.

    Regression test for QR finding: input like read1, read1_1, read1 should not
    create duplicate read1_1 entries. The second read1 should become read1_2.
    """
    input_fasta = tmpdir.join('input.fasta')
    input_fasta.write(
        '>read1\n'
        'ATCG\n'
        '>read1_1\n'
        'GCTA\n'
        '>read1\n'
        'TTTT\n'
        '>read2\n'
        'AAAA\n'
        '>read2_1\n'
        'GGGG\n'
        '>read2_2\n'
        'CCCC\n'
        '>read2\n'
        'NNNN\n'
    )

    output_fasta = str(tmpdir.join('output.fasta'))
    virnucpro_tool._ensure_unique_fasta_ids(str(input_fasta), output_fasta)

    with open(output_fasta, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Extract just the IDs
    ids = [line[1:] for line in lines if line.startswith('>')]

    # Verify all IDs are unique (no collisions)
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    # Verify expected outputs for collision scenarios
    expected_ids = ['read1', 'read1_1', 'read1_2', 'read2', 'read2_1', 'read2_2', 'read2_3']
    assert ids == expected_ids, f"Expected {expected_ids}, got {ids}"


def test_classify_cpu_mode(mocker, virnucpro_tool, test_bam, tmpdir):
    """Test classify() with CPU mode sets CUDA_VISIBLE_DEVICES=-1."""
    mock_popen = mocker.patch('virnucpro.subprocess.Popen', autospec=True)
    mock_process = mock_popen.return_value
    mock_process.communicate.return_value = ('', '')
    mock_process.returncode = 0

    mock_exists = mocker.patch('virnucpro.os.path.exists', return_value=True)
    mock_copy = mocker.patch('virnucpro.shutil.copy')

    out_report = str(tmpdir.join('output.txt'))
    virnucpro_tool.classify(test_bam, out_report, expected_length=500, use_gpu=False)

    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    env = call_args[1]['env']

    assert 'CUDA_VISIBLE_DEVICES' in env
    assert env['CUDA_VISIBLE_DEVICES'] == '-1'


def test_classify_gpu_mode(mocker, virnucpro_tool, test_bam, tmpdir):
    """Test classify() with GPU mode does not override CUDA_VISIBLE_DEVICES."""
    mock_popen = mocker.patch('virnucpro.subprocess.Popen', autospec=True)
    mock_process = mock_popen.return_value
    mock_process.communicate.return_value = ('', '')
    mock_process.returncode = 0

    mock_exists = mocker.patch('virnucpro.os.path.exists', return_value=True)
    mock_copy = mocker.patch('virnucpro.shutil.copy')

    out_report = str(tmpdir.join('output.txt'))
    virnucpro_tool.classify(test_bam, out_report, expected_length=300, use_gpu=True)

    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    env = call_args[1]['env']

    if 'CUDA_VISIBLE_DEVICES' in env:
        assert env['CUDA_VISIBLE_DEVICES'] != '-1'


def test_classify_subprocess_failure(mocker, virnucpro_tool, test_bam, tmpdir):
    """Test classify() raises RuntimeError when subprocess fails."""
    mock_popen = mocker.patch('virnucpro.subprocess.Popen', autospec=True)
    mock_process = mock_popen.return_value
    mock_process.returncode = 1
    mock_process.communicate.return_value = ('', 'Error: something went wrong')

    out_report = str(tmpdir.join('output.txt'))

    with pytest.raises(RuntimeError) as exc_info:
        virnucpro_tool.classify(test_bam, out_report, expected_length=500)

    assert 'VirNucPro failed' in str(exc_info.value)


def test_classify_subprocess_traceback(mocker, virnucpro_tool, test_bam, tmpdir):
    """Test classify() raises RuntimeError when subprocess stderr contains Traceback."""
    mock_popen = mocker.patch('virnucpro.subprocess.Popen', autospec=True)
    mock_process = mock_popen.return_value
    mock_process.returncode = 0
    mock_process.communicate.return_value = ('', 'Traceback (most recent call last):\n  File "prediction.py"')

    out_report = str(tmpdir.join('output.txt'))

    with pytest.raises(RuntimeError) as exc_info:
        virnucpro_tool.classify(test_bam, out_report, expected_length=500)

    assert 'VirNucPro failed' in str(exc_info.value)
    assert 'Traceback' in str(exc_info.value)
