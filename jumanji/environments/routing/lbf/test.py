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

from jumanji.environments import LevelBasedForaging
from jumanji.environments.routing.lbf.generator import RandomGenerator

# Create environment
key = jax.random.key(0)

generator = RandomGenerator(
    grid_size=8,
    fov=8,
    num_agents=3,
    num_food=2,
    max_agent_level=2,
    force_coop=False,
)

# generator(key)

env = LevelBasedForaging(generator=generator)

state1, timestep1 = jax.jit(env.reset)(key)

state1.agents.level = jnp.array([1, 2, 3])
env.render(state1)
plt.savefig("jumanji/environments/routing/lbf/state.png")
action = jnp.array([1, 2, 3])
# state1.food_items.eaten = jnp.array([True, False])

state2, timestep2 = env.step(state1, action)
key, subkey = jax.random.split(key)

env.animate(
    states=[state1, state2],
    interval=100,
    save_path="jumanji/environments/routing/lbf/lbf.gif",
)
Image(filename="jumanji/environments/routing/lbf/lbf.gif", embed=True)
