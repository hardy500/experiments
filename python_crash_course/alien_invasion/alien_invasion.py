import pygame
from settings import Settings
from ship import Ship
from bullet import Bullet
from alien import Alien
from game_stats import GameStats
from buttom import Button
from scoreboard import ScoreBoard

from sys import exit
from time import sleep

class AlienInvasion:
  def __init__(self):
    pygame.init()

    self.clock = pygame.time.Clock()
    self.settings = Settings()
    #self.screen = pygame.display.set_mode((self.settings.screen_height, self.settings.screen_height))
    self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    # Code seeem to work without it
    #self.settings.screen_width = self.screen.get_rect().width
    #self.settings.screen_height = self.screen.get_rect().height
    pygame.display.set_caption("Alien Invasion")

    # Active game state
    self.game_active = False
    self.play_button = Button(self, "Play")

    self.background_color = (self.settings.bg_color)
    self.stats = GameStats(self)
    self.sb = ScoreBoard(self)
    self.ship = Ship(self)
    self.bullets = pygame.sprite.Group()
    self.aliens = pygame.sprite.Group()

    self._create_fleet()

  def run(self):
    while True:
      self._check_event()
      if self.game_active:
        self.ship.update()
        self._update_bullets()
        self._update_aliens()

      self._update_screen()
      self.clock.tick(60)

  def _ship_hit(self):
    if self.stats.ship_left > 0:
      self.stats.ship_left -= 1
      self.sb.prep_ships()

      self.bullets.empty()
      self.aliens.empty()

      self._create_fleet()
      self.ship.center_ship()
      sleep(0.5)
    else:
      self.game_active = False
      pygame.mouse.set_visible(True)

  def _update_aliens(self):
    self._check_fleet_edges()
    self.aliens.update()

    # Look for alien-chip collision
    if pygame.sprite.spritecollideany(self.ship, self.aliens):
      self._ship_hit()

    # Look for aliens hitting the bottom of the screen
    self._check_aliens_bottom()

  def _update_bullets(self):
    # Get rid of bullets that have are beyond the screen
    self.bullets.update()
    for bullet in self.bullets.copy():
      if bullet.rect.bottom <= 0:
        self.bullets.remove(bullet)

    self._check_bullet_alien_collision()

    # Check for any bullets that have hit aliens
    # If so, get rid of the bullet and the alien.
    if not self.aliens:
      self.bullets.empty()
      self._create_fleet()

  def _check_bullet_alien_collision(self):
    collision = pygame.sprite.groupcollide(
        self.bullets,
        self.aliens,
        True,
        True,
    )

    if collision:
      for aliens in collision.values():
        self.stats.score += self.settings.alien_points
      self.sb.prep_score()
      self.sb.check_high_score()

    if not self.aliens:
      self.bullets.empty()
      self._create_fleet()
      self.settings.increase_speed()

      # Increate level
      self.stats.level += 1
      self.sb.prep_level()

  def _check_event(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        exit()
      elif event.type == pygame.MOUSEBUTTONDOWN:
        mouse_pos = pygame.mouse.get_pos()
        self._check_play_button(mouse_pos)
      elif event.type == pygame.KEYDOWN:
        self._check_keydown_events(event)

      elif event.type == pygame.KEYUP:
        self._check_keyup_events(event)

  def _check_play_button(self, mouse_pos):
    button_clicked = self.play_button.rect.collidepoint(mouse_pos)
    if button_clicked and not self.game_active:
      if self.play_button.rect.collidepoint(mouse_pos):
        self.settings.init_dynamic_settings()
        self.stats.reset_stats()
        self.sb.prep_score()
        self.sb.prep_ships()
        self.game_active = True

        # Hide the mouse cursor
        pygame.mouse.set_visible(False)

        # Get rid of any remaining bullets and aliens
        self.bullets.empty()
        self.aliens.empty()

        # Create a new fleet and center the ship
        self._create_fleet()
        self.ship.center_ship()

  def _update_screen(self):
    # Clean screen
    self.screen.fill(self.background_color)
    for bullet in self.bullets.sprites():
      bullet.draw()
    self.ship.draw()
    self.aliens.draw(self.screen)
    self.sb.show()

    # Draw the play button if the game is inactive
    if not self.game_active:
      self.play_button.draw()

    # Make the most recently drawn screen visible
    pygame.display.flip()

  def _check_keydown_events(self, event):
    if event.key == pygame.K_RIGHT:
      self.ship.moving_right = True
    elif event.key == pygame.K_LEFT:
      self.ship.moving_left = True
    elif event.key == pygame.K_UP:
      self.ship.moving_up = True
    elif event.key == pygame.K_DOWN:
      self.ship.moving_down = True
    elif event.key == pygame.K_SPACE:
      self._fire_bullet()
    elif event.key == pygame.K_q:
      exit()

  def _check_keyup_events(self, event):
    if event.key == pygame.K_RIGHT:
      self.ship.moving_right = False
    elif event.key == pygame.K_LEFT:
      self.ship.moving_left = False
    elif event.key == pygame.K_UP:
      self.ship.moving_up = False
    elif event.key == pygame.K_DOWN:
      self.ship.moving_down = False

  def _fire_bullet(self):
    if len(self.bullets) < self.settings.bullet_allowed:
      bullet = Bullet(self)
      self.bullets.add(bullet)

  def _create_fleet(self):
    alien = Alien(self)
    alien_width, alien_height = alien.rect.size

    current_x, current_y = alien_width, alien_height
    while current_y < (self.settings.screen_height - 3*alien_height):
      while current_x < (self.settings.screen_width - 2*alien_width):
        self._create_alien(current_x, current_y)
        current_x += 2 * alien_width

      current_x = alien_width
      current_y += alien_height

  def _create_alien(self, x_pos, y_pos):
    alien = Alien(self)
    alien.x = x_pos
    alien.rect.x = x_pos
    alien.rect.y = y_pos
    self.aliens.add(alien)

  def _check_fleet_edges(self):
    for alien in self.aliens.sprites():
      if alien.check_edges():
        self._change_fleet_direction()
        break

  def _change_fleet_direction(self):
    for alien in self.aliens.sprites():
      alien.rect.y += self.settings.fleed_drop_speed
    self.settings.fleet_direction *= -1

  def _check_aliens_bottom(self):
    for alien in self.aliens.sprites():
      if alien.rect.bottom >= self.settings.screen_height:
        # Treat this the same as if the ship got hit
        self._ship_hit()
        break



if __name__ == "__main__":
  game = AlienInvasion()
  game.run()