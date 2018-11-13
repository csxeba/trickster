from trickster.experience.experience import Experience


class AgentBase:

    def __init__(self, actions, memory: Experience, reward_discount_factor=0.99, state_preprocessor=None):
        self.memory = memory
        self.possible_actions = actions
        self.states = []
        self.rewards = []
        self.actions = []
        self.gamma = reward_discount_factor
        self.learning = True
        self.preprocess = self._preprocess_noop if state_preprocessor is None else state_preprocessor

    def set_learning_mode(self, switch: bool):
        self.learning = switch

    @staticmethod
    def _preprocess_noop(state):
        return state

    def sample(self, state, reward):
        raise NotImplementedError

    def push_experience(self, final_state, final_reward, done=True):
        raise NotImplementedError

    def fit(self, batch_size=32):
        raise NotImplementedError
