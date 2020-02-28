from ..experience import replay_buffer, sampler


class RLAgentBase:

    transition_memory_keys = []
    training_memory_keys = []
    history_keys = []

    def __init__(self,
                 memory_buffer_size: int,
                 separate_training_memory: bool):

        self.transition = replay_buffer.Transition(self.transition_memory_keys)
        self.transition_memory = replay_buffer.Experience(self.transition_memory_keys, max_length=memory_buffer_size)
        if separate_training_memory:
            self.training_memory = replay_buffer.Experience(self.training_memory_keys, max_length=memory_buffer_size)
        else:
            self.training_memory = self.transition_memory
        self.memory_sampler = sampler.ExperienceSampler(self.training_memory)

        self.timestep = 0
        self.episodes = 0
        self.learning = False

    def set_learning_mode(self, switch: bool):
        self.learning = switch

    def sample(self, state, reward, done):
        raise NotImplementedError

    def get_savables(self) -> dict:
        raise NotImplementedError

    def save(self, artifatory_root=None, experiment_name=None, environment_name=None, **metadata):
        import os
        save_root = artifatory_root
        if artifatory_root is None:
            save_root = "../artifactory"
        if experiment_name is not None:
            save_root = os.path.join(save_root, experiment_name)
        if environment_name is not None:
            save_root = os.path.join(save_root, environment_name)
        save_root = os.path.join(save_root, self.__class__.__name__)
        for savable_name, savable in self.get_savables().items():
            meta_suffix = "".join("-{}_{}".format(k, v) for k, v in metadata.items())
            save_path = os.path.join(save_root, "{}{}.h5".format(savable_name, meta_suffix))
            savable.save(save_path)

    def load(self, loadables: dict):
        saveables = self.get_savables()
        for key, value in loadables.items():
            if isinstance(value, str):
                saveables[key].load_weights(value)
            else:
                saveables[key].set_weights(value)

    def dispatch_workers(self, n=1):
        return [self] * n

    def create_worker(self, **worker_kwargs):
        return self

    def end_trajectory(self):
        pass

    def _set_transition(self, state, action, reward, done, **kwargs):
        assert self.learning
        if self.timestep > 0:
            self.transition.set(state_next=state, reward=reward, done=done)
            assert self.transition.ready
            self.transition_memory.store(self.transition)

        self.timestep += 1
        if done:
            self.timestep = 0
            self.episodes += 1
        else:
            self.transition.set(state=state, action=action, **kwargs)

    def fit(self, batch_size=None):
        raise NotImplementedError
