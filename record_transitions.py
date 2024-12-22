"""
This script is used to simply play the Atari games manually.
"""
import imageio
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pygame
from ocatari.core import OCAtari

from argparse import ArgumentParser
from tqdm import tqdm
import os
import pickle as pkl

parser = ArgumentParser()
parser.add_argument("-g", "--game", type=str, default="Pong")
parser.add_argument("-pr", "--print-reward", action="store_true")

args = parser.parse_args()


def save_rgb_array_as_png(rgb_array, filename):
    imageio.imwrite(filename, rgb_array)


class Renderer:
    env: gym.Env

    def __init__(self, env_name: str):
        self.env = OCAtari(env_name, mode="ram", hud=True, render_mode="human",
                           render_oc_overlay=True, frameskip=1)
        self.env.reset()
        self.env.render()  # initialize pygame video system

        self.paused = False
        self.current_keys_down = set()
        self.keys2actions = self.env.unwrapped.get_keys_to_action()
        self.frame = 0
        self.transitions = []
        self.total_frames = 10000
        self.pbar = tqdm(total=self.total_frames)

    def run(self):
        self.running = True
        while self.running:
            self._handle_user_input()
            if not self.paused:
                action = self._get_action()
                obs, reward, term, trunc, info = self.env.step(action)
                self.env.render()
                self.transitions.append(((self.env.get_ram(), self.env.getScreenRGB()), action, reward, term, trunc))
                if term or trunc:
                    self.env.reset()
                self.frame += 1
                self.pbar.update(1)
                if self.frame % self.total_frames == 0:
                    os.makedirs("transitions", exist_ok=True)
                    pkl.dump(self.transitions, open(f"transitions/{args.game}.pkl", "wb"))
        pygame.quit()

    def _get_action(self):
        pressed_keys = list(self.current_keys_down)
        pressed_keys.sort()
        pressed_keys = tuple(pressed_keys)
        if pressed_keys in self.keys2actions.keys():
            return self.keys2actions[pressed_keys]
        else:
            return 0  # NOOP

    def _handle_user_input(self):
        self.current_mouse_pos = np.asarray(pygame.mouse.get_pos())

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:  # window close button clicked
                self.running = False

            elif event.type == pygame.KEYDOWN:  # keyboard key pressed
                if event.key == pygame.K_p:  # 'P': pause/resume
                    self.paused = not self.paused

                if event.key == pygame.K_r:  # 'R': reset
                    self.env.reset()

                elif (event.key,) in self.keys2actions.keys():  # env action
                    self.current_keys_down.add(event.key)

            elif event.type == pygame.KEYUP:  # keyboard key released
                if (event.key,) in self.keys2actions.keys():
                    self.current_keys_down.remove(event.key)


if __name__ == "__main__":
    renderer = Renderer(args.game)
    renderer.run()
