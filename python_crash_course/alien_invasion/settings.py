from dataclasses import dataclass
from typing import Tuple

@dataclass
class Settings:

  def __init__(self):
    # Screen
    self.screen_width = 1200
    self.screen_height =  800
    self.bg_color = (230, 230, 230)
    # Alien settings
    self.alien_speed = 10
    self.fleed_drop_speed = 10

    # Ship
    self.ship_limit = 3

    # Bullet
    self.bullet_speed = 10
    self.bullet_width = 3
    self.bullet_height = 15
    self.bullet_color = (60, 60, 60)
    self.bullet_allowed = 3

    # Score value increase factor
    self.score_scale = 1.5

    # How quickly the game speeds up
    self.speedup_scale = 1.1
    self.init_dynamic_settings()

    # Score settings
    self.alien_points = 50

  def init_dynamic_settings(self):
    self.ship_speed = 1.5
    self.bullet_speed = 2.5
    self.alien_speed = 1.0

    # 1 = right; -1 = left
    self.fleet_direction: int = 1

  def increase_speed(self):
    self.ship_speed *= self.speedup_scale
    self.bullet_speed *= self.speedup_scale
    self.alien_speed *= self.speedup_scale
    self.alien_points = int(self.alien_points * self.score_scale)