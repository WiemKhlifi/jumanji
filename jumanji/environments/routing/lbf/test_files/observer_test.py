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

import jax.numpy as jnp
import pytest

from jumanji.environments.routing.lbf.observer import GridObserver, VectorObserver
from jumanji.environments.routing.lbf.types import Agent, Food, State

# Levels:
# agent grid
# [1, 2, 0],
# [2, 0, 1],
# [0, 0, 0],

# food grid
# [0, 0, 0],
# [0, 4, 0],
# [3, 0, 0],

# IDs:
# agent grid
# [a0, a1, 0],
# [a2, 0, a3],
# [0, 0, 0],

# food grid
# [0, 0, 0],
# [0, f0, 0],
# [f1, 0, 0],


def test_grid_observer(state: State) -> None:
    observer = GridObserver(fov=1, grid_size=3, num_agents=2, num_food=2)
    obs = observer.state_to_observation(state)
    expected_agent_0_view = jnp.array(
        [
            [
                # other agent levels
                [-1, -1, -1],
                [-1, 1, 2],
                [-1, 2, 0],
            ],
            [
                # food levels
                [-1, -1, -1],
                [-1, 0, 0],
                [-1, 0, 4],
            ],
            [
                # access (where can the agent go?)
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ]
    )

    assert jnp.all(obs.agents_view[0, ...] == expected_agent_0_view)
    assert jnp.all(
        obs.action_mask[0, ...] == jnp.array([True, False, False, False, False, True])
    )

    expected_agent_1_view = jnp.array(
        [
            [
                [-1, -1, -1],
                [1, 2, 0],
                [2, 0, 1],
            ],
            [
                [-1, -1, -1],
                [0, 0, 0],
                [0, 4, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 0, 0],
            ],
        ]
    )
    assert jnp.all(obs.agents_view[1, ...] == expected_agent_1_view)
    assert jnp.all(
        obs.action_mask[1, ...] == jnp.array([True, False, True, False, False, True])
    )

    expected_agent_3_view = jnp.array(
        [
            [
                [2, 0, -1],
                [0, 1, -1],
                [0, 0, -1],
            ],
            [
                [0, 0, -1],
                [4, 0, -1],
                [0, 0, -1],
            ],
            [
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
        ]
    )

    assert jnp.all(obs.agents_view[3, ...] == expected_agent_3_view)
    assert jnp.all(
        obs.action_mask[3, ...] == jnp.array([True, True, False, True, False, True])
    )

    # test different fov
    observer = GridObserver(fov=3, grid_size=3, num_agents=2, num_food=2)
    # test eaten food is not visible
    eaten = jnp.array([True, False])
    food = Food(
        eaten=eaten,
        id=state.food_items.id,
        position=state.food_items.position,
        level=state.food_items.level,
    )
    state = state.replace(food=food)  # type: ignore

    obs = observer.state_to_observation(state)
    expected_agent_2_view = jnp.array(
        [
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 1, 2, 0, -1],
                [-1, -1, -1, 2, 0, 1, -1],
                [-1, -1, -1, 0, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 0, 0, 0, -1],
                [-1, -1, -1, 0, 0, 0, -1],
                [-1, -1, -1, 3, 0, 0, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    assert jnp.all(obs.agents_view[2, ...] == expected_agent_2_view)
    assert jnp.all(
        obs.action_mask[2, ...] == jnp.array([True, False, True, False, False, True])
    )


def test_vector_observer(state: State) -> None:
    observer = VectorObserver(fov=1, grid_size=3, num_agents=2, num_food=2)
    obs = observer.state_to_observation(state)
    expected_agent_0_view = jnp.array(
        [1, 1, 4, -1, -1, 0, 0, 0, 1, 0, 1, 2, 1, 0, 2, -1, -1, 0]
    )
    assert jnp.all(obs.agents_view[0, ...] == expected_agent_0_view)
    assert jnp.all(
        obs.action_mask[0, ...] == jnp.array([True, False, False, False, False, True])
    )

    expected_agent_2_view = jnp.array(
        [1, 1, 4, 2, 0, 3, 1, 0, 2, 0, 0, 1, 0, 1, 2, -1, -1, 0]
    )
    assert jnp.all(obs.agents_view[2, ...] == expected_agent_2_view)
    assert jnp.all(
        obs.action_mask[2, ...] == jnp.array([True, False, False, False, False, True])
    )

    # test different fov
    observer = VectorObserver(fov=3, grid_size=3, num_agents=2, num_food=2)
    # test eaten food is not visible
    eaten = jnp.array([True, False])
    food = Food(
        eaten=eaten,
        id=state.food_items.id,
        position=state.food_items.position,
        level=state.food_items.level,
    )
    state = state.replace(food=food)  # type: ignore

    obs = observer.state_to_observation(state)
    expected_agent_3_view = jnp.array(
        [-1, -1, 0, 2, 0, 3, 1, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 2]
    )
    assert jnp.all(obs.agents_view[3, ...] == expected_agent_3_view)
    assert jnp.all(
        obs.action_mask[3, ...] == jnp.array([True, True, False, True, True, True])
    )


# TODO: add test for the "2s" case.


# Define constants for tests
FOV = 5
GRID_SIZE = 10
NUM_AGENTS = 3
NUM_FOOD = 5
MAX_AGENT_LEVEL = 10
MAX_FOOD_LEVEL = 5
TIME_LIMIT = 100

# Define helper functions to create test entities
def create_agent(id, position, level):
    return Agent(id=id, position=position, level=level)


def create_food(position, level, eaten):
    return Food(position=position, level=level, eaten=eaten)


def create_state(agents, food_items, step_count):
    return State(agents=agents, food_items=food_items, step_count=step_count)


@pytest.mark.parametrize(
    "agent_id, agent_position, agent_level, food_positions, food_levels, food_eaten, expected_observation",
    [
        # Test ID: 1 - Happy path test with single agent and single food
        (
            0,
            (2, 2),
            1,
            [(2, 3)],
            [3],
            [False],
            jnp.array([2, 3, 3, 2, 2, 1, -1, -1, 0, -1, -1, 0, -1, -1, 0]),
        ),
        # Test ID: 2 - Edge case with agent at the edge of the grid
        (
            0,
            (0, 0),
            1,
            [(0, 1)],
            [3],
            [False],
            jnp.array([0, 1, 3, 0, 0, 1, -1, -1, 0, -1, -1, 0, -1, -1, 0]),
        ),
        # Test ID: 3 - Error case with food out of bounds (should be handled gracefully)
        (
            0,
            (2, 2),
            1,
            [(GRID_SIZE + 1, GRID_SIZE + 1)],
            [3],
            [False],
            jnp.array([-1, -1, 0, 2, 2, 1, -1, -1, 0, -1, -1, 0, -1, -1, 0]),
        ),
        # Additional test cases should be added here for 100% coverage
    ],
    ids=["happy-single-agent-food", "edge-agent-edge-grid", "error-food-out-of-bounds"],
)
def test_make_observation(
    agent_id,
    agent_position,
    agent_level,
    food_positions,
    food_levels,
    food_eaten,
    expected_observation,
):
    # Arrange
    observer = VectorObserver(
        fov=FOV, grid_size=GRID_SIZE, num_agents=NUM_AGENTS, num_food=NUM_FOOD
    )
    agent = create_agent(agent_id, agent_position, agent_level)
    food_items = [
        create_food(pos, level, eaten)
        for pos, level, eaten in zip(food_positions, food_levels, food_eaten)
    ]
    state = create_state(agents=[agent], food_items=food_items, step_count=0)

    # Act
    observation = observer.make_observation(agent, state)

    # Assert
    chex.assert_equal(observation, expected_observation)


# Additional tests for other methods like transform_positions, extract_foods_info, extract_agents_info, compute_action_mask, state_to_observation, and observation_spec should be added following the same pattern.
