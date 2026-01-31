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


@pytest.fixture
def paired_end_bam(tmpdir):
    """Create a test BAM file with paired-end reads."""
    bam_path = str(tmpdir.join('paired.bam'))
    header = {'HD': {'VN': '1.0'}}
    with pysam.AlignmentFile(bam_path, 'wb', header=header) as bam:
        # First read of pair
        read1 = pysam.AlignedSegment()
        read1.query_name = 'readA'
        read1.query_sequence = 'ATCG' * 10
        read1.flag = 65  # paired, first in pair
        read1.reference_id = -1
        read1.reference_start = -1
        read1.mapping_quality = 0
        read1.cigar = None
        read1.next_reference_id = -1
        read1.next_reference_start = -1
        read1.template_length = 0
        read1.query_qualities = pysam.qualitystring_to_array('I' * 40)
        bam.write(read1)

        # Second read of pair
        read2 = pysam.AlignedSegment()
        read2.query_name = 'readA'
        read2.query_sequence = 'GCTA' * 10
        read2.flag = 129  # paired, second in pair
        read2.reference_id = -1
        read2.reference_start = -1
        read2.mapping_quality = 0
        read2.cigar = None
        read2.next_reference_id = -1
        read2.next_reference_start = -1
        read2.template_length = 0
        read2.query_qualities = pysam.qualitystring_to_array('I' * 40)
        bam.write(read2)

    return bam_path


def test_bam_to_fasta_paired_end_suffix(virnucpro_tool, paired_end_bam, tmpdir):
    """Test _bam_to_fasta() adds /1 and /2 suffixes for paired-end reads."""
    out_fasta = str(tmpdir.join('output.fasta'))
    virnucpro_tool._bam_to_fasta(paired_end_bam, out_fasta)

    with open(out_fasta, 'r') as f:
        content = f.read()

    # Should have /1 and /2 suffixes
    assert '>readA/1\n' in content
    assert '>readA/2\n' in content


def test_bam_to_fasta_unpaired(virnucpro_tool, test_bam, tmpdir):
    """Test _bam_to_fasta() does not add suffix for unpaired reads."""
    out_fasta = str(tmpdir.join('output.fasta'))
    virnucpro_tool._bam_to_fasta(test_bam, out_fasta)

    with open(out_fasta, 'r') as f:
        content = f.read()

    # Unpaired reads should not have /1 or /2
    assert '>read0\n' in content
    assert '>read1\n' in content
    assert '>read2\n' in content
    assert '/1' not in content
    assert '/2' not in content


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
