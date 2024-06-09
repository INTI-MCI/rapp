import pytest

from rapp import cli


def test_polarimeter(monkeypatch, tmp_path):
    cli.run([
        'polarimeter',
        '--work-dir', str(tmp_path),
        '--mock-esp',
        '--mock-adc',
        '--hwp-delay-position', '0'
    ])

    with pytest.raises(SystemExit):
        cli.run([])
