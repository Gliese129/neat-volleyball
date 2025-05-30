import math

from slime_volleyball.core import constants
from slime_volleyball.core import utils
from slime_volleyball.core.objects import (
    RelativeState,
    half_circle,
    circle,
)


class Agent:
    """keeps track of the agent in the game. note this is not the policy network"""

    def __init__(self, dir, x, y, c):  # standardize_actions=False):
        self.dir = dir  # -1 means left, 1 means right player for symmetry.
        self.x = x
        self.y = y
        self.r = 1.5
        self.c = c
        self.vx = 0
        self.vy = 0
        self.desired_vx = 0
        self.desired_vy = 0
        self.state = RelativeState()
        self.emotion = "happy"
        # hehe...
        self.life = constants.MAXLIVES

    def lives(self):
        return self.life

    def set_action(self, action):
        forward = False
        backward = False
        jump = False

        if action[0] > 0:
            forward = True
        if action[1] > 0:
            backward = True
        if action[2] > 0:
            jump = True

        self.desired_vx = 0
        self.desired_vy = 0

        if forward and (not backward):
            self.desired_vx = -constants.PLAYER_SPEED_X
        if backward and (not forward):
            self.desired_vx = constants.PLAYER_SPEED_X
        if jump:
            self.desired_vy = constants.PLAYER_SPEED_Y

    def move(self):
        self.x += self.vx * constants.TIMESTEP
        self.y += self.vy * constants.TIMESTEP

    def step(self):
        self.x += self.vx * constants.TIMESTEP
        self.y += self.vy * constants.TIMESTEP

    def update(self):
        self.vy += constants.GRAVITY * constants.TIMESTEP

        if self.y <= constants.REF_U + constants.NUDGE * constants.TIMESTEP:
            self.vy = self.desired_vy

        self.vx = self.desired_vx * self.dir

        self.move()

        if self.y <= constants.REF_U:
            self.y = constants.REF_U
            self.vy = 0

        # stay in their own half:
        if self.x * self.dir <= (constants.REF_WALL_WIDTH / 2 + self.r):
            self.vx = 0
            self.x = self.dir * (constants.REF_WALL_WIDTH / 2 + self.r)

        if self.x * self.dir >= (constants.REF_W / 2 - self.r):
            self.vx = 0
            self.x = self.dir * (constants.REF_W / 2 - self.r)

    def update_state(self, ball, opponent):
        """normalized to side, appears different for each agent's perspective"""
        # agent's self
        self.state.x = self.x * self.dir
        self.state.y = self.y
        self.state.vx = self.vx * self.dir
        self.state.vy = self.vy
        # ball
        self.state.bx = ball.x * self.dir
        self.state.by = ball.y
        self.state.bvx = ball.vx * self.dir
        self.state.bvy = ball.vy
        # opponent
        self.state.ox = opponent.x * (-self.dir)
        self.state.oy = opponent.y
        self.state.ovx = opponent.vx * (-self.dir)
        self.state.ovy = opponent.vy

    def get_observation(self):
        return self.state.get_observation()

    def display(self, canvas, bx, by):
        x = self.x
        y = self.y
        r = self.r

        angle = math.pi * 60 / 180
        if self.dir == 1:
            angle = math.pi * 120 / 180
        eyeX = 0
        eyeY = 0

        canvas = half_circle(
            canvas, utils.toX(x), utils.toY(y), utils.toP(r), color=self.c
        )

        # track ball with eyes (replace with observed info later):
        c = math.cos(angle)
        s = math.sin(angle)
        ballX = bx - (x + (0.6) * r * c)
        ballY = by - (y + (0.6) * r * s)

        if self.emotion == "sad":
            ballX = -self.dir
            ballY = -3

        dist = math.sqrt(ballX * ballX + ballY * ballY)
        eyeX = ballX / dist
        eyeY = ballY / dist

        canvas = circle(
            canvas,
            utils.toX(x + (0.6) * r * c),
            utils.toY(y + (0.6) * r * s),
            utils.toP(r) * 0.3,
            color=(255, 255, 255),
        )
        canvas = circle(
            canvas,
            utils.toX(x + (0.6) * r * c + eyeX * 0.15 * r),
            utils.toY(y + (0.6) * r * s + eyeY * 0.15 * r),
            utils.toP(r) * 0.1,
            color=(0, 0, 0),
        )

        # draw coins (lives) left
        for i in range(1, self.life):
            canvas = circle(
                canvas,
                utils.toX(self.dir * (constants.REF_W / 2 + 0.5 - i * 2.0)),
                constants.WINDOW_HEIGHT - utils.toY(1.5),
                utils.toP(0.5),
                color=constants.COIN_COLOR,
            )

        return canvas
