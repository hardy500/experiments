import pygame

class Button:
  def __init__(self, game, msg):
    self.screen = game.screen
    self.screen_rect = game.screen.get_rect()

    # Set the dimension and properties of the button
    self.width, self.height = 200, 50
    self.button_color = (0, 135, 0)
    self.text_color = (255, 255, 255)
    self.font = pygame.font.SysFont(None, 48)

    # Build the bottom's rect object and center it
    self.rect = pygame.Rect(0, 0, self.width, self.height)
    self.rect.center = self.screen_rect.center

    # The button message need the be prepped only once
    self._prep_msg(msg)

  def _prep_msg(self, msg):
    self.msg_image = self.font.render(msg, True, self.text_color, self.button_color)
    self.msg_image_rect = self.msg_image.get_rect()
    self.msg_image_rect.center = self.rect.center

  def draw(self):
    self.screen.fill(self.button_color, self.rect)
    self.screen.blit(self.msg_image, self.msg_image_rect)