from gym_minigrid.extendedminigrid import *
from gym_minigrid.register import register

class DeadEndWaterEnv(ExMiniGridEnv):
    """
    Unsafe grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Generate the wall which separate the rooms
        j = 2
        while j < height-2 :
            i = 2
            while i < width-1 :
                self.grid.set(i,j, Wall())
                i += 1
            j += 2

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Place water
        self.grid.set(width -2, height - 3, Water())

        # Set start position
        self.start_pos = (1, 1)
        self.start_dir = 0

        self.mission = "get to the green goal square without going into the deadend"

    def step(self,action):
        # Reset if agent step on water without knowing it
        if action == self.actions.forward and self.worldobj_in_agent(1,0) == "water" :
            return self.gen_obs(), 0, True, "died"
        else:
            return super().step(action)

class DeadEndWaterEnv6x6(DeadEndWaterEnv):
    def __init__(self):
        super().__init__(size=6)

class DeadEndWaterEnv16x16(DeadEndWaterEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-DeadEndWaterEnv-6x6-v0',
    entry_point='gym_minigrid.envs:DeadEndWaterEnv6x6'
)

register(
    id='MiniGrid-DeadEndWaterEnv-8x8-v0',
    entry_point='gym_minigrid.envs:DeadEndWaterEnv'
)

register(
    id='MiniGrid-DeadEndWaterEnv-16x16-v0',
    entry_point='gym_minigrid.envs:DeadEndWaterEnv16x16'
)
