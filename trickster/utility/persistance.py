import pathlib
from typing import Union


def save_agent(agent, output_directory: Union[pathlib.Path, str], model_name_suffix: str):
    output_directory = pathlib.Path(output_directory)
    for name, savable in agent.get_savables().items():
        save_path = output_directory / "_".join([name, model_name_suffix]) + ".h5"
        savable.save(save_path)
