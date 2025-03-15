import torch
import pygame
import random
import numpy as np
from collections import namedtuple
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet

pygame.init()

Point = namedtuple("Point", "x, y")

YELLOW = (255, 255, 102)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (50, 153, 213)

BLOCK_SIZE = 20
FRAMESPEED = 40


class SnakeAgent:
    def __init__(self):
        self.model = Linear_QNet(11, 256, 3)
        self.model.load_state_dict(torch.load("model/model.pth"))
        self.model.eval()

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()

        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move


def GameLoop(agent):
    game = SnakeGameAI()
    clock = pygame.time.Clock()

    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)

        if done:
            print(f"Final Score: {score}")
            break

        clock.tick(FRAMESPEED)

    return score


def KeepWindowOpen():
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
    pygame.quit()


agent = SnakeAgent()

print("Running Snake AI Inference...")
score = GameLoop(agent)
KeepWindowOpen()
