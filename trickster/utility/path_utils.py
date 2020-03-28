import os

from artifactorium import Artifactorium


def ensure_workdir():
    cwd = os.path.split(os.getcwd())[-1]
    if cwd in ["examples", "experiments", "test"]:
        os.chdir("..")


class Artifactory(Artifactorium):

    __slots__ = "tensorboard", "logs", "render"

    def __init__(self, root="default", experiment_name=None):
        if root == "default":
            ensure_workdir()
            root = os.path.abspath("artifactory/")
        super().__init__(root, experiment_name)
        self.register_path("tensorboard", "NOW", "tensorboard")
        self.register_path("logs", "NOW", "logs")
        self.register_path("render", "NOW", "renders")
