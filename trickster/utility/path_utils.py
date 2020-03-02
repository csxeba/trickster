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
            root = os.path.abspath("artifactory/")
        self.root = root
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def make_logdir(self, experiment_name=""):
        path = os.path.join(self.root, experiment_name, self.now, "logs")
        os.makedirs(path, exist_ok=True)
        return path


defaults = _DefaultPaths()
