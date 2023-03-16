class GameStats:
  def __init__(self, game):
    self.settings = game.settings
    self.reset_stats()

    # High score should never be reset
    self.high_score = 0

  def reset_stats(self):
    self.ship_left = self.settings.ship_limit
    self.score = 0
    self.level = 1