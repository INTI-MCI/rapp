import re

REGEX_NUMBER_AFTER_WORD = r"(?<={word})-?\d+(?:\.\d+)?"


def parse_input_parameters_from_filepath(filepath):
    cycles_find = re.findall(REGEX_NUMBER_AFTER_WORD.format(word="cycles"), filepath)
    step_find = re.findall(REGEX_NUMBER_AFTER_WORD.format(word="step"), filepath)
    samples_find = re.findall(REGEX_NUMBER_AFTER_WORD.format(word="samples"), filepath)

    cycles = cycles_find[0] if cycles_find else 0
    step = step_find[0] if step_find else 0
    samples = samples_find[0] if samples_find else 0

    return (cycles, step, samples)
