import time
import logging

from rapp.polarimeter import run as main_polarimeter
from rapp.motion_controller import ESP301Error


LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def setup_logger():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # Hide logging from external libraries
    external_libs = ['matplotlib', 'PIL']
    for lib in external_libs:
        logging.getLogger(lib).setLevel(logging.ERROR)


def main():
    setup_logger()
    e1 = dict(cycles=1, step=1, samples=128, reps=50, velocity=4, acceleration=4, deceleration=4)
    e2 = dict(cycles=1, step=1, samples=128, reps=12, velocity=4, acceleration=4, deceleration=4,
              hwp_enabled=True,  hwp_cycles=0.0014, hwp_step=0.1, hwp_delay_position=0.5)
    e3 = dict(cycles=1, step=1, samples=128, reps=12, velocity=4, acceleration=4, deceleration=4,
              hwp_enabled=True,  hwp_cycles=0.14, hwp_step=10, hwp_delay_position=0.5)
    e4 = dict(cycles=1, step=20, samples=128, reps=1, velocity=5, acceleration=4, deceleration=4)
            #  hwp_enabled=True, hwp_cycles=0.1, hwp_step=0.1, hwp_delay_position=0.5)
    e5 = dict(cycles=1, step=1, samples=134, reps=20, velocity=4, acceleration=4, deceleration=4)
    e6 = dict(cycles=1, step=1, samples=169, reps=50, velocity=2)
    experimentos = [e2, e3]
    tiempo_inicial_de_espera = 30  # tiempo de espera en segundos
    parametros_en_dirname = False

    logger.info("Inicio: tiempo de espera de {} segundos".format(tiempo_inicial_de_espera))
    time.sleep(tiempo_inicial_de_espera)  # tiempo de espera en segundos
    logger.info("Fin del tiempo de espera. Iniciando experimentos...")
    # basename = "phase-diff-with-quartz-perkin-elmer-retrorreflexion"  # Acordarse de guardar!!
    basename = "riao2026-hwp-cs1237-2"  # Acordarse de guardar!!
    for i, exp in enumerate(experimentos, 1):
        logger.info("EXPERIMENTO {}: {}".format(i, exp))
        prefix = f"{basename}-{i}"
        try:
            if parametros_en_dirname:
                for k, v in exp.items():
                    prefix += f"{k}-{v}"
            main_polarimeter(prefix=prefix, **exp)
        except ESP301Error as e:
            logger.warning(f"Found error: {e}. Retrying...")
            time.sleep(30)
            main_polarimeter(prefix=prefix, **exp)


if __name__ == '__main__':
    main()
