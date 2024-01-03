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

import chex
import jax
import jax.numpy as jnp

from jumanji.environments.routing.lbf.generator import RandomGenerator


def is_adj(pos0: chex.Array, pos1: chex.Array) -> chex.Array:
    return jnp.linalg.norm(pos0 - pos1) == 1


def test_generator() -> None:
    key = jax.random.PRNGKey(42)

    grid_size = 6
    fov = 6
    num_agents = 7
    num_food = 6
    max_agent_level = 10
    force_coop = False

    gen = RandomGenerator(
        grid_size=grid_size,
        fov=fov,
        num_agents=num_agents,
        num_food=num_food,
        max_agent_level=max_agent_level,
        force_coop=force_coop,
    )
    state = gen(key)

    # Test food and agents placed within grid bounds.
    assert jnp.all(state.agents.position >= 0)
    assert jnp.all(state.agents.position < grid_size)
    assert jnp.all(state.food_items.position >= 0)
    assert jnp.all(state.food_items.position < grid_size)

    # test no foods are adjacent to each other
    adjaciencies = jax.vmap(jax.vmap(is_adj, in_axes=(0, None)), in_axes=(None, 0))(
        state.food_items.position, state.food_items.position
    )
    assert jnp.all(~adjaciencies)

    # test no foods are on the edge of the grid
    assert jnp.all(state.food_items.position != 0)
    assert jnp.all(state.food_items.position != grid_size - 1)

    # test that food levels aren't too high to be eaten
    assert jnp.all(state.food_items.level <= jnp.sum(state.agents.level))
    assert jnp.all(state.food_items.level >= num_food)
