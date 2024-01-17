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

# TODO:
# ? add 2s case
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


# Test cases for VectorObserver class
def test_lbf_observer_initialization(lbf_env_2s):
    observer = VectorObserver(fov=2, grid_size=8, num_agents=2, num_food=2)
    assert observer.fov == lbf_env_2s.fov
    assert observer.grid_size == lbf_env_2s.grid_size
    assert observer.num_agents == lbf_env_2s.num_agents
    assert observer.num_food == lbf_env_2s.num_food


# TODO test_action_mask():


def test_vector_full_obs_items_info(agent0, agents, food_items):
    observer_full_obs = VectorObserver(fov=6, grid_size=6, num_agents=3, num_food=3)

    visible_agents1 = jnp.all(
        jnp.abs(agent0.position - agents.position) <= observer_full_obs.fov,
        axis=-1,
    )

    visible_foods2 = (
        jnp.all(
            jnp.abs(agent0.position - food_items.position) <= observer_full_obs.fov,
            axis=-1,
        )
        & ~food_items.eaten
    )

    (
        agent_i_infos,
        agent_xs,
        agent_ys,
        agent_levels,
    ) = observer_full_obs.extract_agents_info(agent0, visible_agents1, agents)
    results_agents = jnp.stack([agent_xs, agent_ys, agent_levels], axis=1)
    food_xs, food_ys, food_levels = observer_full_obs.extract_foods_info(
        agent0, visible_foods2, food_items
    )
    results_food = jnp.stack([food_xs, food_ys, food_levels], axis=1)

    agent_i_expected, expected_results_agents = jnp.array([0, 0, 1]), jnp.stack(
        [jnp.array([1, 2]), jnp.array([1, 2]), jnp.array([2, 3])], axis=1
    )
    expected_results_food = (jnp.array([[2, 1], [2, 3], [4, 2]]), jnp.array([4, 5, 3]))
    assert agent_i_infos == agent_i_expected
    assert jnp.all(jnp.allclose(results_agents, expected_results_agents))
    assert jnp.all(jnp.allclose(results_food, expected_results_food))


def test_vector_full_obs_items_info(agent0, agents, food_items):
    observer_partial_obs = VectorObserver(fov=2, grid_size=6, num_agents=3, num_food=3)

    visible_agents1 = jnp.all(
        jnp.abs(agent0.position - agents.position) <= observer_partial_obs.fov,
        axis=-1,
    )

    visible_foods2 = (
        jnp.all(
            jnp.abs(agent0.position - food_items.position) <= observer_partial_obs.fov,
            axis=-1,
        )
        & ~food_items.eaten
    )

    results_agents = observer_partial_obs.extract_agents_info(
        agent0, visible_agents1, agents
    )
    results_food = observer_partial_obs.extract_foods_info(
        agent0, visible_foods2, food_items
    )
    expected_results_agents = (
        jnp.array([0, 0, 1]),
        jnp.array([1, 2]),
        jnp.array([1, 2]),
        jnp.array([2, 3]),
    )
    expected_results_food = (jnp.array([[2, 1], [2, 3], [4, 2]]), jnp.array([4, 5, 3]))
    assert jnp.all(jnp.allclose(results_agents, expected_results_agents))
    assert jnp.all(jnp.allclose(results_food, expected_results_food))


# def test_grid_observer(state: State) -> None:
#     observer = GridObserver(fov=1, grid_size=3, num_agents=2, num_food=2)
#     obs = observer.state_to_observation(state)
#     expected_agent_0_view = jnp.array(
#         [
#             [
#                 # other agent levels
#                 [-1, -1, -1],
#                 [-1, 1, 2],
#                 [-1, 2, 0],
#             ],
#             [
#                 # food levels
#                 [-1, -1, -1],
#                 [-1, 0, 0],
#                 [-1, 0, 4],
#             ],
#             [
#                 # access (where can the agent go?)
#                 [0, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 0],
#             ],
#         ]
#     )

#     assert jnp.all(obs.agents_view[0, ...] == expected_agent_0_view)
#     assert jnp.all(
#         obs.action_mask[0, ...] == jnp.array([True, False, False, False, False, True])
#     )

#     expected_agent_1_view = jnp.array(
#         [
#             [
#                 [-1, -1, -1],
#                 [1, 2, 0],
#                 [2, 0, 1],
#             ],
#             [
#                 [-1, -1, -1],
#                 [0, 0, 0],
#                 [0, 4, 0],
#             ],
#             [
#                 [0, 0, 0],
#                 [0, 1, 1],
#                 [0, 0, 0],
#             ],
#         ]
#     )
#     assert jnp.all(obs.agents_view[1, ...] == expected_agent_1_view)
#     assert jnp.all(
#         obs.action_mask[1, ...] == jnp.array([True, False, True, False, False, True])
#     )

#     expected_agent_3_view = jnp.array(
#         [
#             [
#                 [2, 0, -1],
#                 [0, 1, -1],
#                 [0, 0, -1],
#             ],
#             [
#                 [0, 0, -1],
#                 [4, 0, -1],
#                 [0, 0, -1],
#             ],
#             [
#                 [0, 1, 0],
#                 [0, 1, 0],
#                 [1, 1, 0],
#             ],
#         ]
#     )

#     assert jnp.all(obs.agents_view[3, ...] == expected_agent_3_view)
#     assert jnp.all(
#         obs.action_mask[3, ...] == jnp.array([True, True, False, True, False, True])
#     )

#     # test different fov
#     observer = GridObserver(fov=3, grid_size=3, num_agents=2, num_food=2)
#     # test eaten food is not visible
#     eaten = jnp.array([True, False])
#     food = Food(
#         eaten=eaten,
#         id=state.food_items.id,
#         position=state.food_items.position,
#         level=state.food_items.level,
#     )
#     state = state.replace(food=food)  # type: ignore

#     obs = observer.state_to_observation(state)
#     expected_agent_2_view = jnp.array(
#         [
#             [
#                 [-1, -1, -1, -1, -1, -1, -1],
#                 [-1, -1, -1, -1, -1, -1, -1],
#                 [-1, -1, -1, 1, 2, 0, -1],
#                 [-1, -1, -1, 2, 0, 1, -1],
#                 [-1, -1, -1, 0, 0, 0, -1],
#                 [-1, -1, -1, -1, -1, -1, -1],
#                 [-1, -1, -1, -1, -1, -1, -1],
#             ],
#             [
#                 [-1, -1, -1, -1, -1, -1, -1],
#                 [-1, -1, -1, -1, -1, -1, -1],
#                 [-1, -1, -1, 0, 0, 0, -1],
#                 [-1, -1, -1, 0, 0, 0, -1],
#                 [-1, -1, -1, 3, 0, 0, -1],
#                 [-1, -1, -1, -1, -1, -1, -1],
#                 [-1, -1, -1, -1, -1, -1, -1],
#             ],
#             [
#                 [0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 1, 0],
#                 [0, 0, 0, 1, 1, 0, 0],
#                 [0, 0, 0, 0, 1, 1, 0],
#                 [0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0],
#             ],
#         ]
#     )
#     assert jnp.all(obs.agents_view[2, ...] == expected_agent_2_view)
#     assert jnp.all(
#         obs.action_mask[2, ...] == jnp.array([True, False, True, False, False, True])
#     )


# def test_vector_observer_full_obs(state: State) -> None:
#     observer = VectorObserver(fov=6, grid_size=6, num_agents=3, num_food=3)
#     obs = observer.state_to_observation(state)
#     expected_agent_0_view = jnp.array(
#         [1, 1, 4, -1, -1, 0, 0, 0, 1, 0, 1, 2, 1, 0, 2, -1, -1, 0]
#     )
#     assert jnp.all(obs.agents_view[0, ...] == expected_agent_0_view)
#     assert jnp.all(
#         obs.action_mask[0, ...] == jnp.array([True, False, False, False, False, True])
#     )

#     expected_agent_2_view = jnp.array(
#         [1, 1, 4, 2, 0, 3, 1, 0, 2, 0, 0, 1, 0, 1, 2, -1, -1, 0]
#     )
#     assert jnp.all(obs.agents_view[2, ...] == expected_agent_2_view)
#     assert jnp.all(
#         obs.action_mask[2, ...] == jnp.array([True, False, False, False, False, True])
#     )

#     # test different fov
#     observer = VectorObserver(fov=3, grid_size=3, num_agents=2, num_food=2)
#     # test eaten food is not visible
#     eaten = jnp.array([True, False])
#     food = Food(
#         eaten=eaten,
#         id=state.food_items.id,
#         position=state.food_items.position,
#         level=state.food_items.level,
#     )
#     state = state.replace(food=food)

#     obs = observer.state_to_observation(state)
#     expected_agent_3_view = jnp.array(
#         [-1, -1, 0, 2, 0, 3, 1, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 2]
#     )
#     assert jnp.all(obs.agents_view[3, ...] == expected_agent_3_view)
#     assert jnp.all(
#         obs.action_mask[3, ...] == jnp.array([True, True, False, True, True, True])
#     )
