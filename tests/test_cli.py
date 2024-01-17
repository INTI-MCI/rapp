import pytest

from rapp import cli


class AnimationMock:
    def start(self, *args, **kwargs):
        pass


def test_polarimeter(monkeypatch, tmp_path):
    # monkeypatch.setattr(LoadedStringAnimation, 'build', lambda *args, **kwargs: AnimationMock())

    cli.run([
        'polarimeter', '--work-dir', str(tmp_path), '--mock-esp', '--mock-adc', '--hwp-delay', '0'
    ])

    with pytest.raises(SystemExit):
        cli.run([])

    # with pytest.raises(AssertionError):
    #    cli.run(['loaded_string'])
