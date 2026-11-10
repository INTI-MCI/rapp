import logging

import numpy as np
import matplotlib.pyplot as plt

from rapp import adc
from rapp import measurement
from rapp import constants as ct
from rapp.analysis.plot import Plot
from rapp.simulations import simulation


logger = logging.getLogger(__name__)

TPL_LOG = "angle={}°, φerr: {}."
TPL_LABEL = "cycles={}\nstep={}°\nsamples={}\nreps={}"
TPL_FILENAME = "sim_error_vs_phase-reps-{}-cycles-{}-step-{}-samples-{}.{}"
TPL_FILENAME_COMP = "sim_error_vs_phase-comparison-{}-{}.{}"


def run(
    folder,
    angle_range=(-90, 90),
    n_angles=30,
    method="DFT",
    cycles=1,
    step=1,
    samples=5,
    reps=1,
    max_v=adc.MAXV,
    dynamic_range=0.7,
    n_bits=(adc.BITS, 24),
    k=0,
    noise=True,
    show=False,
    save=True,
):
    print("")
    logger.info("PHASE DIFFERENCE VS ANGLE")

    angles = np.linspace(*angle_range, num=n_angles)
    amplitude = (max_v * dynamic_range) / 2

    errors = {}
    for bits in n_bits:
        logger.info("Method: {}, bits={}, reps={}".format(method, bits, reps))

        errors[bits] = []
        for angle in angles:
            n_res = simulation.n_simulations(
                N=reps,
                angle=angle,
                method=method,
                allow_nan=True,
                cycles=cycles,
                step=step,
                samples=samples,
                A=amplitude,
                max_v=max_v,
                bits=bits,
                a0_noise=measurement.A0_NOISE if noise else None,
                a1_noise=measurement.A1_NOISE if noise else None,
                a0_k=k,
            )

            error = n_res.rmse()
            errors[bits].append(error)

            logger.info(TPL_LOG.format(angle, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_PHASE_DIFF,
        ysci=True,
        yoom=-3,
        # xint=True,
        folder=folder,
    )

    for bits, (method, plot_config) in zip(n_bits, simulation.METHODS.items()):
        plot.add_data(angles, errors[bits], label=f"bits={bits}", **plot_config)

    plot.legend(loc="upper right", fontsize=13)

    plot.the_ax.axhline(y=0, lw=1, ls="dashed")

    annotation = TPL_LABEL.format(cycles, step, samples, reps)
    plot.the_ax.text(0.05, 0.7, annotation, transform=plot.the_ax.transAxes)
    yfmt = simulation.get_axis_formatter(power_limits=(-3, -3))
    plot.the_ax.yaxis.set_major_formatter(yfmt)

    plot.the_ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    max_value = max(max(x) for x in errors.values())
    plot.the_ax.set_ylim(-1e-3, max_value + 0.005)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(filename=TPL_FILENAME.format(reps, cycles, step, samples, format_))

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return angles, errors

def run_comparison(
    folder,
    angle_range=(0, 30),
    n_angles=4,
    method="NLS",
    #max_v=adc.MAXV,
    dynamic_range=0.7,
    k=0,
    noise=True,
    show=False,
    save=True,
):
    print("")
    logger.info("ADC COMPARISON: PHASE DIFFERENCE VS ANGLE")

    angles = np.linspace(*angle_range, num=n_angles)
    #amplitude = (max_v * dynamic_range) / 2

    errors = {}

    for adc_name, config in adc.SUPPORTED_ADCS.items():

        bits = config["BITS"]
        sample_rate = config["SAMPLE_RATE"]
        max_v = config["GAINS"][1][0]

        amplitude = (max_v * dynamic_range) / 2

        sim_config = config["SIMULATION"]

        cycles = sim_config["CYCLES"]
        step = sim_config["STEP"]
        reps = sim_config["REPS"]
        samples = sim_config["SAMPLES"]
        noise_rms = config["NOISE_RMS"] if noise else 0
        print(f"{adc_name}: adc_noise = {noise_rms:.2e} V")

        logger.info(
            "ADC: %s, sample rate: %s SPS, bits: %s",
            adc_name, sample_rate, bits
        )

        errors[adc_name] = []

        for angle in angles:

            n_res = simulation.n_simulations(
                N=reps,
                angle=angle,
                method=method,
                allow_nan=True,
                cycles=cycles,
                step=step,
                samples=samples,
                A=amplitude,
                max_v=max_v,
                bits=bits,
                a0_noise=measurement.A0_NOISE if noise else None,
                a1_noise=measurement.A1_NOISE if noise else None,
                a0_k=k,
                adc_noise=noise_rms,
            )

            error = n_res.rmse()
            errors[adc_name].append(error)

            logger.info(TPL_LOG.format(angle, "{:.2E}".format(error)))

    plot = Plot(
        ylabel=ct.LABEL_PHI_ERR,
        xlabel=ct.LABEL_PHASE_DIFF,
        ysci=True,
        #yoom=-3,
        folder=folder,
    )
    plot.the_ax.set_title(
        f"ADC comparison: {angle_range[0]}° to {angle_range[1]}° "
        f"(step: {(angle_range[1] - angle_range[0]) / (n_angles - 1):g}°) "
        f"– Method: {method}"
    )

    for adc_name, values in errors.items():
        print(f"\n{adc_name}")
        print(np.asarray(values))
        print(f"Mínimo: {np.nanmin(values):.6e}")
        print(f"Máximo: {np.nanmax(values):.6e}")

    for adc_name, config in adc.SUPPORTED_ADCS.items():
        samples = config["SIMULATION"]["SAMPLES"]

        plot.add_data(
            angles,
            errors[adc_name],
            marker="o",
            linestyle="-",
            label=f"{adc_name} ({samples} samples)",
        )

    plot.legend(loc="best", fontsize=13)
    plot.the_ax.axhline(y=0, lw=1, ls="dashed")

    #yfmt = simulation.get_axis_formatter(power_limits=(-3, -3))
    #plot.the_ax.yaxis.set_major_formatter(yfmt)
    plot.the_ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )

    plot.the_ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    max_value = max(max(values) for values in errors.values())
    margin = max_value * 0.05
    plot.the_ax.set_ylim(0, max_value + margin)

    if save:
        for format_ in simulation.FORMATS:
            plot.save(
                filename=f"sim_error_vs_phase-comparison-{method}.{format_}"
            )

    if show:
        plot.show()

    plot.close()

    logger.info("Done.")

    return angles, errors