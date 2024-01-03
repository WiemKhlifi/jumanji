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

import abc
from typing import Tuple

import chex
import jax
from jax import numpy as jnp

from jumanji.environments.routing.lbf.types import Agent, Food, State


# TODO: do we need this base class, there is only one viable generator to match lbf?
class Generator(abc.ABC):
    """Base class for generators for the LBF environment."""

    def __init__(
        self,
        grid_size: int,
        fov: int,
        num_agents: int,
        num_food: int,
        max_agent_level: int,
        force_coop: bool = False,
    ) -> None:
        """Initialises a LBF generator, used to generate grids for the LBF environment.

        Args:
            grid_size: size of the grid to generate.
            fov: field of view of an agent.
            num_agents: number of agents on the grid.
            num_food: number of food items on the grid.
            max_agent_level: maximum level of the agents (inclusive).
            force_coop: Force cooperation between agents.
        """
        # Add assertions to check the validity of the input values.
        assert 5 <= grid_size < 20, "Grid size must be between 5 and 19."
        assert 0 < fov <= grid_size, "Field of view must be greater than 0."
        assert 1 <= num_agents < 20, "Number of agents must be between 1 and 19."
        assert 1 <= num_food < 10, "Number of food items must be between 1 and 10."
        assert max_agent_level > 0, "Maximum agent level must be greater than 0."

        if fov is None:
            fov = grid_size
        self.grid_size = grid_size
        self.fov = fov
        self.num_agents = num_agents
        self.num_food = num_food
        self.max_agent_level = max_agent_level
        self.force_coop = force_coop

    @abc.abstractmethod
    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `LBF` state that contains the grid and the agents' layout.

        Returns:
            A `LBF` state.
        """


class RandomGenerator(Generator):
    """Randomly generates `LBF` grids.

    If too many food itms and agents are given for a small grid size, it might not be able to
    place them on the grid. Because of jax this will fail silently and many food items will
    be placed at (0, 0).
    """

    def __init__(
        self,
        grid_size: int,
        fov: int,
        num_agents: int,
        num_food: int,
        max_agent_level: int,
        force_coop: bool,
    ) -> None:
        """Initialises an lbf generator, used to generate grids for
        the LevelBasedForaging environment."""
        super().__init__(
            grid_size, fov, num_agents, num_food, max_agent_level, force_coop
        )

    def sample_food(self, key: chex.PRNGKey) -> chex.Array:
        """Randomly samples food positions on the grid, ensuring no two food items are adjacent
        and no food is placed on the edge of the grid.

        Args:
            key (chex.PRNGKey): The random key for reproducible randomness.

        Returns:
            chex.Array: An array containing the flat indices of food on the grid.
                        Each element corresponds to the flattened position of a food item.
        """
        flat_size = self.grid_size**2
        pos_keys = jax.random.split(key, self.num_food)

        # Create a mask to exclude edges
        mask = jnp.ones(flat_size, dtype=bool)
        mask = mask.at[jnp.arange(self.grid_size)].set(False)  # top
        mask = mask.at[jnp.arange(flat_size - self.grid_size, flat_size)].set(
            False
        )  # bottom
        mask = mask.at[jnp.arange(0, flat_size, self.grid_size)].set(False)  # left
        mask = mask.at[jnp.arange(self.grid_size - 1, flat_size, self.grid_size)].set(
            False
        )  # right

        def take_positions(
            mask: chex.Array, key: chex.PRNGKey
        ) -> Tuple[chex.Array, chex.Array]:
            food_flat_pos = jax.random.choice(key=key, a=flat_size, shape=(), p=mask)

            # Mask out adjacent positions to avoid placing food items next to each other
            adj_positions = jnp.array(
                [
                    food_flat_pos,
                    food_flat_pos + 1,  # right
                    food_flat_pos - 1,  # left
                    food_flat_pos + self.grid_size,  # up
                    food_flat_pos - self.grid_size,  # down
                ]
            )

            return mask.at[adj_positions].set(False), food_flat_pos

        _, food_flat_positions = jax.lax.scan(take_positions, mask, pos_keys)

        # Unravel indices to get the 2D coordinates (x, y)
        food_positions_x, food_positions_y = jnp.unravel_index(
            food_flat_positions, (self.grid_size, self.grid_size)
        )
        food_positions = jnp.stack([food_positions_x, food_positions_y], axis=1)

        return food_positions

    def sample_agents(self, key: chex.PRNGKey, mask: chex.Array) -> chex.Array:
        """Randomly samples agent positions on the grid, avoiding positions occupied by food.

        Args:
            key (chex.PRNGKey): The random key.
            mask (chex.Array): The mask of the grid where 1s correspond to empty cells
            and 0s to food cells.

        Returns:
            chex.Array: An array containing the positions of agents on the grid.
                        Each row corresponds to the (x, y) coordinates of an agent.
        """
        agent_flat_positions = jax.random.choice(
            key=key,
            a=self.grid_size**2,
            shape=(self.num_agents,),
            replace=False,  # Avoid agent positions overlaping
            p=mask,
        )
        # Unravel indices to get x and y coordinates
        agent_positions_x, agent_positions_y = jnp.unravel_index(
            agent_flat_positions, (self.grid_size, self.grid_size)
        )

        # Stack x and y coordinates to form a 2D array
        agent_positions = jnp.stack([agent_positions_x, agent_positions_y], axis=1)

        return agent_positions

    def sample_levels(
        self, max_level: int, shape: chex.Shape, key: chex.PRNGKey
    ) -> chex.Array:
        """Randomly samples levels within the specified shape.

        Args:
            max_level (int): The maximum level (inclusive).
            shape (chex.Shape): The shape of the array to be generated.
            key (chex.PRNGKey): The random key.

        Returns:
            chex.Array: An array containing randomly sampled levels.
        """
        return jax.random.randint(
            key,
            shape=shape,
            minval=1,
            maxval=max_level + 1,
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `LBF` state containing the grid and the agents' layout.

        Args:
            key (chex.PRNGKey): The random key for reproducible randomness.

        Returns:
            State: A `LBF` state containing information about the grid, agents, and food items.
        """

        (
            food_pos_key,
            food_level_key,
            agent_pos_key,
            agent_level_key,
            key,
        ) = jax.random.split(key, 5)

        # Generate positions for food items
        food_positions = self.sample_food(food_pos_key)

        # Generate positions for agents. The mask contains 0's where food is placed,
        # 1's where agents can be placed.
        mask = jnp.ones((self.grid_size, self.grid_size), dtype=bool)
        mask = mask.at[food_positions].set(False)
        mask = mask.ravel()
        agent_positions = self.sample_agents(key=agent_pos_key, mask=mask)

        # Generate levels for agents and food items
        agent_levels = self.sample_levels(
            self.max_agent_level, (self.num_agents,), agent_level_key
        )
        max_food_level = jnp.sum(
            jnp.sort(agent_levels)[:3]
        )  # In the worst case, 3 agents are needed to eat a food item

        # Determine food levels based on the maximum level of agents
        food_levels = jnp.where(
            self.force_coop,
            jnp.full(shape=(self.num_food,), fill_value=max_food_level),
            self.sample_levels(max_food_level, (self.num_food,), food_level_key),
        )

        # Create pytrees for agents and food items
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            position=agent_positions,
            level=agent_levels,
        )
        food_items = jax.vmap(Food)(
            id=jnp.arange(self.num_food),
            position=food_positions,
            level=food_levels,
        )
        step_count = jnp.array(0, jnp.int32)

        return State(
            key=key, step_count=step_count, agents=agents, food_items=food_items
        )
