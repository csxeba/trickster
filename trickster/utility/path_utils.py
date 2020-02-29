import datetime
import os


def ensure_workdir():
    cwd = os.path.split(os.getcwd())[-1]
    if cwd in ["examples", "experiments", "test"]:
        os.chdir("..")


class _DefaultPaths:

    def __init__(self, root="default"):
        if root == "default":
            ensure_workdir()
            root = os.path.abspath("artifactory/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.root = root

    def _build_and_check_path(self, sfx):
        path = os.path.join(self.root, sfx)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def logdir(self):
        return self._build_and_check_path("logs")


defaults = _DefaultPaths()
