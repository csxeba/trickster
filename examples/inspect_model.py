from keras.models import load_model

from trickster import Policy, Rollout, RolloutConfig

from grund.reskiv import ReskivConfig, Reskiv
from grund.util.movement import get_movement_vectors
from grund.util.screen import CV2Screen

NUM_MOVES = 4
MOVES = get_movement_vectors(num_directions=NUM_MOVES)

rcfg = ReskivConfig(canvas_shape=[64, 64, 3], player_radius=3, target_size=3)
env = Reskiv(rcfg)

actor = load_model("../models/reskiv/reinforce_latest.h5")
agent = Policy(actor, actions=MOVES)
screen = CV2Screen(scale=2)
rollout = Rollout(agent, env, RolloutConfig(learning=False, screen=screen))

while 1:
    rollout.reset()
    rollout.rollout(verbose=1)
