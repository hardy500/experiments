import pygame
from settings import Settings
from ship import Ship

from sys import exit

class AlienInvasion:
  def __init__(self):
    pygame.init()

    self.clock = pygame.time.Clock()
    self.settings = Settings()

    self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    # Code seeem to work without it
    #self.settings.screen_width = self.screen.get_rect().width
    #self.settings.screen_height = self.screen.get_rect().height
    pygame.display.set_caption("Alien Invasion")

    self.background_color = (self.settings.bg_color)
    self.ship = Ship(self)

  def run(self):
    while True:
      self._check_event()
      self.ship.update()
      self._update_screen()
      self.clock.tick(60)

  def _check_event(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        exit()
      elif event.type == pygame.KEYDOWN:
        self._check_keydown_events(event)

      elif event.type == pygame.KEYUP:
        self._check_keyup_events(event)

  def _update_screen(self):
    # Clean screen
    self.screen.fill(self.background_color)
    self.ship.draw()

    # Make the most recently drawn screen visible
    pygame.display.flip()

  def _check_keydown_events(self, event):
    if event.key == pygame.K_RIGHT:
      self.ship.moving_right = True
    elif event.key == pygame.K_LEFT:
      self.ship.moving_left = True
    elif event.key == pygame.K_q:
      exit()

  def _check_keyup_events(self, event):
    if event.key == pygame.K_RIGHT:
      self.ship.moving_right = False
    elif event.key == pygame.K_LEFT:
      self.ship.moving_left = False





if __name__ == "__main__":
  ai = AlienInvasion()
  ai.run()

