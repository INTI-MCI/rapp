import logging

import pytest

from rapp import cli


def test_polarimeter(tmp_path):
    cli.run([
        'polarimeter',
        '--work-dir', str(tmp_path),
        '--mock-esp',
        '--mock-adc',
        '--mock-pm100',
        '--hwp-delay-position', '0'
    ])

    with pytest.raises(SystemExit):
        cli.run([])


if __name__ == '__main__':
    tmp_path = '/tmp/pytest-of-pablo/pytest-0/test_run1'
    test_polarimeter(tmp_path)


def test_verbose_enables_debug_logging():
    app = cli.RAPP(["plot_raw", "--verbose", "data/table-riao.csv"])

    assert app._logging_level == logging.DEBUG
