"""Tests for VirNucPro CLI."""
import os
import sys
from unittest import mock

import pytest

import virnucpro_cli


def test_cli_help(capsys):
    """Verify --help displays usage and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        with mock.patch('sys.argv', ['virnucpro_cli.py', '--help']):
            virnucpro_cli.parse_args()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert 'VirNucPro: Classify viral sequences' in captured.out
    assert 'input_bam' in captured.out
    assert 'output_tsv' in captured.out


def test_cli_version(capsys, monkeypatch):
    """Verify --version displays env var value."""
    monkeypatch.setenv('VIRNUCPRO_VERSION', '1.2.3-test')

    with pytest.raises(SystemExit) as exc_info:
        with mock.patch('sys.argv', ['virnucpro_cli.py', '--version']):
            virnucpro_cli.parse_args()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert '1.2.3-test' in captured.out


def test_cli_basic_invocation():
    """Verify basic invocation calls VirNucPro.classify() with correct args."""
    with mock.patch('virnucpro.VirNucPro') as mock_virnucpro:
        mock_instance = mock.MagicMock()
        mock_virnucpro.return_value = mock_instance

        with mock.patch('sys.argv', ['virnucpro_cli.py', 'input.bam', 'output.tsv']):
            virnucpro_cli.main()

        mock_virnucpro.assert_called_once_with(virnucpro_path=None)
        mock_instance.classify.assert_called_once_with(
            'input.bam',
            'output.tsv',
            expected_length=500,
            use_gpu=None
        )


def test_cli_expected_length():
    """Verify --expected-length parameter is passed correctly."""
    with mock.patch('virnucpro.VirNucPro') as mock_virnucpro:
        mock_instance = mock.MagicMock()
        mock_virnucpro.return_value = mock_instance

        with mock.patch('sys.argv', ['virnucpro_cli.py', 'input.bam', 'output.tsv', '--expected-length', '300']):
            virnucpro_cli.main()

        mock_instance.classify.assert_called_once_with(
            'input.bam',
            'output.tsv',
            expected_length=300,
            use_gpu=None
        )


def test_cli_missing_args():
    """Verify error message on missing positional args."""
    with pytest.raises(SystemExit) as exc_info:
        with mock.patch('sys.argv', ['virnucpro_cli.py']):
            virnucpro_cli.parse_args()

    assert exc_info.value.code == 2


def test_cli_conflicting_gpu_flags():
    """Verify error on --use-gpu + --no-gpu."""
    with mock.patch('virnucpro.VirNucPro'):
        with pytest.raises(SystemExit) as exc_info:
            with mock.patch('sys.argv', ['virnucpro_cli.py', 'input.bam', 'output.tsv', '--use-gpu', '--no-gpu']):
                virnucpro_cli.main()

        assert exc_info.value.code == 1


def test_cli_use_gpu_flag():
    """Verify --use-gpu sets gpu_mode=True."""
    with mock.patch('virnucpro.VirNucPro') as mock_virnucpro:
        mock_instance = mock.MagicMock()
        mock_virnucpro.return_value = mock_instance

        with mock.patch('sys.argv', ['virnucpro_cli.py', 'input.bam', 'output.tsv', '--use-gpu']):
            virnucpro_cli.main()

        mock_instance.classify.assert_called_once_with(
            'input.bam',
            'output.tsv',
            expected_length=500,
            use_gpu=True
        )


def test_cli_no_gpu_flag():
    """Verify --no-gpu sets gpu_mode=False."""
    with mock.patch('virnucpro.VirNucPro') as mock_virnucpro:
        mock_instance = mock.MagicMock()
        mock_virnucpro.return_value = mock_instance

        with mock.patch('sys.argv', ['virnucpro_cli.py', 'input.bam', 'output.tsv', '--no-gpu']):
            virnucpro_cli.main()

        mock_instance.classify.assert_called_once_with(
            'input.bam',
            'output.tsv',
            expected_length=500,
            use_gpu=False
        )


def test_cli_virnucpro_path():
    """Verify --virnucpro-path parameter is passed correctly."""
    with mock.patch('virnucpro.VirNucPro') as mock_virnucpro:
        mock_instance = mock.MagicMock()
        mock_virnucpro.return_value = mock_instance

        with mock.patch('sys.argv', ['virnucpro_cli.py', 'input.bam', 'output.tsv', '--virnucpro-path', '/custom/path']):
            virnucpro_cli.main()

        mock_virnucpro.assert_called_once_with(virnucpro_path='/custom/path')


def test_cli_classify_exception():
    """Verify exit code 1 when classify() raises."""
    with mock.patch('virnucpro.VirNucPro') as mock_virnucpro:
        mock_instance = mock.MagicMock()
        mock_instance.classify.side_effect = RuntimeError("Test error")
        mock_virnucpro.return_value = mock_instance

        with pytest.raises(SystemExit) as exc_info:
            with mock.patch('sys.argv', ['virnucpro_cli.py', 'input.bam', 'output.tsv']):
                virnucpro_cli.main()

        assert exc_info.value.code == 1
