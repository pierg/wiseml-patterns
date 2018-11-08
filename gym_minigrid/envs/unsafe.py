from gym_minigrid.extendedminigrid import *
from gym_minigrid.register import register


class UnsafeEnv(ExMiniGridEnv):
    """
    Unsafe grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def step(self, action):
        # Reset if agent step on water without knowing it
        if action == self.actions.forward and self.worldobj_in_agent(1, 0) == "water":
            return self.gen_obs(), 0, True, "died"
        else:
            return super().step(action)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place water
        self.grid.set(width - 5, height - 2, Water())

        # Set start position
        self.start_pos = (1, 1)
        self.start_dir = 0

        self.mission = "get to the green goal square without moving on water"


class UnsafeEnv6x6(UnsafeEnv):
    def __init__(self):
        super().__init__(size=6)


class UnsafeEnv16x16(UnsafeEnv):
    def __init__(self):
        super().__init__(size=16)


register(
    id='MiniGrid-Unsafe-6x6-v0',
    entry_point='gym_minigrid.envs:UnsafeEnv6x6'
)

register(
    id='MiniGrid-Unsafe-8x8-v0',
    entry_point='gym_minigrid.envs:UnsafeEnv'
)

register(
    id='MiniGrid-Unsafe-16x16-v0',
    entry_point='gym_minigrid.envs:UnsafeEnv16x16'
)
