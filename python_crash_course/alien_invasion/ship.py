import pygame
from pygame.sprite import Sprite
from sys import exit

class Ship(Sprite):
  def __init__(self, game):
    super().__init__()
    self.screen = game.screen
    self.screen_rect = game.screen.get_rect()
    self.settings = game.settings

    # Load image and get its rect
    self.image = pygame.image.load('images/ship.bmp')
    self.rect = self.image.get_rect()

    # Start each nwe ship at the bottom center of the screen
    self.rect.midbottom = self.screen_rect.midbottom
    self.x = float(self.rect.x) # horizontal position
    self.y = float(self.rect.y) # vertical position

    self.moving_right = False
    self.moving_left = False
    self.moving_up = False
    self.moving_down = False

  def draw(self):
    # Draw the ship at its current location
    self.screen.blit(self.image, self.rect)

  def update(self):
    if self.moving_right and self.rect.right < self.screen_rect.right:
      self.x += self.settings.ship_speed
    if self.moving_left and self.rect.left > 0:
      self.x -= self.settings.ship_speed

    if self.moving_up and self.rect.top > 0:
      self.y -= self.settings.ship_speed
    if self.moving_down and self.rect.bottom < self.screen_rect.bottom:
      self.y += self.settings.ship_speed

    # Update rect object
    self.rect.x = self.x
    self.rect.y = self.y

  def center_ship(self):
    self.rect.midbottom = self.screen_rect.midbottom
    self.x = float(self.rect.x)