import torch
import numpy as np
import pygame
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet

device = torch.device("cpu")


class InferenceAgent:
    def __init__(self, model_path):
        self.model = Linear_QNet(11, 256, 3)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
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
        state0 = torch.tensor(np.array(state), dtype=torch.float).to(device)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move


def run_game(model_path):
    agent = InferenceAgent(model_path)
    game = SnakeGameAI()

    while True:
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)

        if done:
            print(f"Game Over! Final Score: {score}")
            break


if __name__ == "__main__":
    try:
        model_path = "model/model.pth"
        run_game(model_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        pygame.quit()
