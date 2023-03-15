import pygame

class AlienInvasion:
  def __init__(self, screen_width=800, screen_height=600):
    pygame.init()

    self.screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Alien Invasion")

    self.background_color = (255, 255, 255)

  def run(self):
    running = True
    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False

      # Clean screen
      self.screen.fill(self.background_color)

      # Make the most recently drawn screen visible
      pygame.display.flip()


if __name__ == "__main__":
  ai = AlienInvasion()
  ai.run()

