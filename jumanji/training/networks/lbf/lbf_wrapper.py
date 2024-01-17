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

from typing import Tuple

import chex
import jax.numpy as jnp

from jumanji.environments.routing.lbf import LevelBasedForaging
from jumanji.environments.routing.lbf.types import State
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper


class LbfWrapper(Wrapper):
    """
    Multi-agent wrapper for the Level-Based Foraging environment.
    """

    def __init__(self, env: LevelBasedForaging) -> None:
        super().__init__(env)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep(timestep)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep:
        """Aggregate individual rewards across agents."""

        # This aggregation was adopted from the EpyMARL wrapper script:
        # https://github.com/uoe-agents/epymarl/blob/main/src/envs/__init__.py#L116
        team_reward = jnp.sum(timestep.reward)
        # All agent's discount is the same
        team_discount = timestep.discount[0]

        return timestep.replace(reward=team_reward, discount=team_discount)
