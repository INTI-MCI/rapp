from rapp import polarimeter


def test_main(tmp_path):
    polarimeter.main(
        cycles=0,
        samples=1,
        delay_position=0,
        hwp_delay=0,
        mock_esp=True,
        mock_adc=True,
        overwrite=True,
        work_dir=tmp_path,
        # plot=True
    )
