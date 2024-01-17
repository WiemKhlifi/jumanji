# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from IPython.display import Image

import jumanji.environments.routing.lbf.utils as utils
from jumanji.environments import LevelBasedForaging
from jumanji.environments.routing.lbf.generator import RandomGenerator

# Create environment
key = jax.random.key(0)

generator = RandomGenerator(
    grid_size=6,
    fov=2,
    num_agents=3,
    num_food=2,
    max_agent_level=2,
    force_coop=False,
)

env = LevelBasedForaging(generator=generator, grid_observation=True)

for _ in range(2):
    state1, timestep1 = jax.jit(env.reset)(key)
    state1.agents.level = jnp.array([1, 2, 3])
    state1.food_items.level = jnp.array([7, 3])
    state1.agents.position = jnp.array([[5, 0], [4, 1], [5, 2]])
    # env.render(state1)

    actions = jnp.array([5, 5, 5])
    state2, timestep2 = env.step(state1, actions)
    # env.render(state2)
    key, subkey = jax.random.split(key)

plt.savefig("jumanji/environments/routing/lbf/state.png")

env.animate(
    states=[state1, state2],
    interval=100,
    save_path="jumanji/environments/routing/lbf/lbf.gif",
)
Image(filename="jumanji/environments/routing/lbf/lbf.gif", embed=True)
