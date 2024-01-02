from rapp import polarimeter


def test_polarimeter(tmp_path):
    polarimeter.main(
        cycles=0,
        samples=1,
        delay_position=0,
        test_esp=True,
        test_adc=True,
        overwrite=True,
        work_dir=tmp_path
    )
