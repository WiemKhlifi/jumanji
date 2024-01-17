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
import pytest

from jumanji.environments.routing.lbf.generator import RandomGenerator
from jumanji.environments.routing.lbf.types import State


def are_entities_adjacent(pos0: chex.Array, pos1: chex.Array) -> chex.Array:
    return jnp.linalg.norm(pos0 - pos1) == 1


def test_generator() -> None:
    key = jax.random.PRNGKey(42)

    grid_size = 6
    fov = 6
    num_agents = 7
    num_food = 6
    max_agent_level = 10
    force_coop = False

    generator = RandomGenerator(
        grid_size=grid_size,
        fov=fov,
        num_agents=num_agents,
        num_food=num_food,
        max_agent_level=max_agent_level,
        force_coop=force_coop,
    )
    state = generator(key)

    # Test food and agents placed within grid bounds.
    assert jnp.all(state.agents.position >= 0)
    assert jnp.all(state.agents.position < grid_size)
    assert jnp.all(state.food_items.position >= 0)
    assert jnp.all(state.food_items.position < grid_size)

    # test no foods are adjacent to each other
    adjaciencies = jax.vmap(
        jax.vmap(are_entities_adjacent, in_axes=(0, None)), in_axes=(None, 0)
    )(state.food_items.position, state.food_items.position)
    assert jnp.all(~adjaciencies)

    # test no foods are on the edge of the grid
    assert jnp.all(state.food_items.position != 0)
    assert jnp.all(state.food_items.position != grid_size - 1)

    # test that food levels aren't too high to be eaten
    assert jnp.all(state.food_items.level <= jnp.sum(state.agents.level))
    assert jnp.all(state.food_items.level >= num_food)


# Test IDs for parametrization
HAPPY_PATH_ID = "happy_path"
EDGE_CASE_TOO_MANY_AGENTS_ID = "edge_case_too_many_agents"
EDGE_CASE_TOO_MANY_FOOD_ID = "edge_case_too_many_food"
ERROR_CASE_INVALID_GRID_SIZE_ID = "error_case_invalid_grid_size"


@pytest.mark.parametrize(
    "test_id, grid_size, fov, num_agents, num_food, max_agent_level, force_coop, expected_exception",
    [
        # Happy path tests with various realistic test values
        (HAPPY_PATH_ID, 5, 2, 2, 3, 2, False, None),
        (HAPPY_PATH_ID, 10, 3, 5, 10, 3, True, None),
        # Edge cases
        # Too many agents for the grid size, might result in an infinite loop or all agents at (0, 0)
        (EDGE_CASE_TOO_MANY_AGENTS_ID, 3, 1, 10, 2, 2, False, None),
        # Too many food items for the grid size, might result in many food items at (0, 0)
        (EDGE_CASE_TOO_MANY_FOOD_ID, 3, 1, 2, 10, 2, False, None),
        # Error cases
        # Invalid grid size (too small)
        (ERROR_CASE_INVALID_GRID_SIZE_ID, 1, 1, 1, 1, 1, False, ValueError),
    ],
)
def test_random_generator(
    test_id,
    grid_size,
    fov,
    num_agents,
    num_food,
    max_agent_level,
    force_coop,
    expected_exception,
):
    key = jax.random.PRNGKey(0)

    # Arrange
    if expected_exception:
        with pytest.raises(expected_exception):
            RandomGenerator(
                grid_size, fov, num_agents, num_food, max_agent_level, force_coop
            )
        return
    else:
        generator = RandomGenerator(
            grid_size, fov, num_agents, num_food, max_agent_level, force_coop
        )

    # Act
    state = generator(key)

    # Assert
    assert isinstance(
        state, State
    ), f"Test ID {test_id}: The state should be an instance of State."
    assert (
        state.agents.shape[0] == num_agents
    ), f"Test ID {test_id}: The number of agents should match the input."
    assert (
        state.food_items.shape[0] == num_food
    ), f"Test ID {test_id}: The number of food items should match the input."
    if not force_coop:
        assert jnp.all(
            state.food_items.level <= jnp.sum(jnp.sort(state.agents.level)[:3])
        ), f"Test ID {test_id}: Food levels should be less than or equal to the sum of the lowest three agent levels."
    else:
        assert jnp.all(
            state.food_items.level == jnp.sum(jnp.sort(state.agents.level)[:3])
        ), f"Test ID {test_id}: Food levels should be equal to the sum of the lowest three agent levels when force_coop is True."
    assert jnp.all(
        state.agents.level <= max_agent_level
    ), f"Test ID {test_id}: Agent levels should be less than or equal to max_agent_level."
    assert jnp.all(state.agents.position >= 0) and jnp.all(
        state.agents.position < grid_size
    ), f"Test ID {test_id}: Agent positions should be within the grid."
    assert jnp.all(state.food_items.position >= 0) and jnp.all(
        state.food_items.position < grid_size
    ), f"Test ID {test_id}: Food positions should be within the grid."
    assert (
        state.step_count == 0
    ), f"Test ID {test_id}: Initial step count should be zero."
