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

# flake8: noqa: CCR001

import math
from typing import Callable, Optional, Sequence, Tuple

import chex
import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from numpy.typing import NDArray

import jumanji
import jumanji.environments.routing.lbf.constants as constants
from jumanji.environments.routing.lbf.types import Agent, Entity, Food, State
from jumanji.tree_utils import tree_slice
from jumanji.viewer import Viewer


class LevelBasedForagingViewer(Viewer):
    def __init__(
        self,
        grid_size: Tuple[int, int],
        name: str = "LevelBasedForaging",
        render_mode: str = "human",
    ) -> None:
        """Viewer for the LevelBasedForaging environment.

        Args:
            grid_size: the size of the grid (width, height)
            name: custom name for the Viewer. Defaults to `LevelBasedForaging`.
        """
        self._name = name
        self.rows = self.cols = grid_size
        self.grid_size = 50

        self.cell_size = self.grid_size
        self.icon_size = self.cell_size / 3
        self.adjust_center = self.cell_size / 2

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)

        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[animation.Animation] = None

    def render(self, state: State) -> Optional[NDArray]:
        """Render the given state of the `LevelBasedForaging` environment.

        Args:
            state: the environment state to render.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._prepare_figure(ax)
        self._draw_state(ax, state)
        return self._display(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig = plt.figure(f"{self._name}Animation", figsize=constants._FIGURE_SIZE)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax = fig.add_subplot(111)
        plt.close(fig)
        self._prepare_figure(ax)

        def make_frame(state: State) -> None:
            ax.clear()
            self._prepare_figure(ax)
            self._draw_state(ax, state)

        # Create the animation object.
        self._animation = animation.FuncAnimation(
            fig,
            make_frame,
            frames=states,
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        plt.close(self._name)

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, figsize=constants._FIGURE_SIZE, facecolor="black")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        if recreate:
            fig.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot(111)
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _prepare_figure(self, ax: plt.Axes) -> None:
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.patch.set_alpha(0.0)
        ax.set_axis_off()

        ax.set_aspect("equal", "box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    def _draw_state(self, ax: plt.Axes, state: State) -> None:
        self._draw_grid(ax)
        self._draw_foods(state.foods, ax)
        self._draw_agents(state.agents, ax)

    def _draw_grid(self, ax: plt.Axes) -> None:
        """Draw grid of warehouse floor."""
        lines = []
        # VERTICAL LINES
        for r in range(self.rows + 1):
            lines.append(
                [
                    (0, (self.grid_size + 1) * r + 1),
                    ((self.grid_size + 1) * self.cols, (self.grid_size + 1) * r + 1),
                ]
            )

        # HORIZONTAL LINES
        for c in range(self.cols + 1):
            lines.append(
                [
                    ((self.grid_size + 1) * c + 1, 0),
                    ((self.grid_size + 1) * c + 1, (self.grid_size + 1) * self.rows),
                ]
            )

        lc = LineCollection(lines, colors=(1, 1, 1))
        ax.add_collection(lc)

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self._name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

    def _draw_agents(self, agents: Agent, ax: plt.Axes) -> None:
        """Draw the agents on the grid."""
        num_agents = len(agents.level)

        for i in range(num_agents):
            agent = tree_slice(agents, i)
            cell_center = self._entity_position(agent)

            # Read the image file
            img = mpimg.imread("icons/agent.png")

            # Resize the image
            # img_resized = resize(img, (self.icon_size, self.icon_size))

            # Create an OffsetImage and add it to the axis
            imagebox = OffsetImage(img, zoom=self.icon_size / self.cell_size)
            ab = AnnotationBbox(
                imagebox, (cell_center[0], cell_center[1]), frameon=False, zorder=10
            )
            ax.add_artist(ab)

            # Add a rectangle (polygon) next to the agent with the agent's level
            self.draw_badge(agent, ax, cell_center)

    def _draw_foods(self, foods: Food, ax: plt.Axes) -> None:
        """Draw the foods on the grid."""
        num_foods = len(foods.level)

        for i in range(num_foods):
            food = tree_slice(foods, i)
            if food.eaten:
                continue

            # Read the image file
            img = mpimg.imread("icons/apple.png")
            cell_center = self._entity_position(food)

            # Create an OffsetImage and add it to the axis
            imagebox = OffsetImage(img, zoom=self.icon_size / self.cell_size)
            ab = AnnotationBbox(
                imagebox, (cell_center[0], cell_center[1]), frameon=False, zorder=10
            )
            ax.add_artist(ab)

            # Add a rectangle (polygon) next to the agent with the food's level
            self.draw_badge(food, ax, cell_center)

    def _entity_position(self, entity: Entity) -> Tuple[float, float]:
        """Return the position of an entity on the grid."""
        row, col = entity.position
        return (
            (row + 1) * self.cell_size + row - self.adjust_center,
            (col + 1) * self.cell_size + col - self.adjust_center,
        )

    def draw_badge(
        self, entity: Entity, ax: plt.Axes, anchor_point: Tuple[float, float]
    ) -> None:
        #   resolution = 6
        #   radius = self.grid_size / 5

        #   badge_x = anchor_point[0] * self.grid_size + (3 / 4) * self.grid_size
        #   badge_y = self.height - self.grid_size * (anchor_point[1] + 1) + (1 / 4) * self.grid_size

        # make a circle
        #   verts = []
        #   for i in range(resolution):
        #       angle = 2 * np.pi * i / resolution
        #       x = radius * np.cos(angle) + badge_x
        #       y = radius * np.sin(angle) + badge_y
        #       verts += [[x, y]]

        #       circle = plt.Polygon(
        #               verts,
        #               edgecolor="white",
        #               facecolor="white",
        #           )

        # ax.add_patch(circle)
        # Calculate the center of the rectangle
        center_x = anchor_point[0] + self.cell_size / 3
        center_y = anchor_point[1] - self.cell_size / 3

        rectangle = plt.Rectangle(
            xy=(center_x, center_y),
            width=self.cell_size / 3,
            height=self.cell_size / 3,
            edgecolor="white",
            facecolor="black",
            zorder=10,  # Adjust zorder to ensure the rectangle is drawn below images
        )
        ax.add_patch(rectangle)

        ax.annotate(
            str(entity.level),
            xy=(center_x + 10, center_y + 10),
            color="white",
            ha="center",
            va="center",
            zorder=12,
        )
