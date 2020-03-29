import os

from artifactorium import Artifactorium as _Artifactorium


def ensure_workdir():
    cwd = os.path.split(os.getcwd())[-1]
    if cwd in ["examples", "experiments", "test"]:
        os.chdir("..")


class Artifactory(_Artifactorium):

    __slots__ = "tensorboard", "logs", "renders"
    _default = None

    def __init__(self, root, experiment_name=None):
        super().__init__(root, experiment_name, "NOW")
        self.register_path("tensorboard")
        self.register_path("logs")
        self.register_path("renders")

    @classmethod
    def make_default(cls, experiment_name=None):
        if cls._default is None:
            ensure_workdir()
            root = os.path.abspath("artifactory/")
            cls._default = cls(root, experiment_name)
        return cls._default
