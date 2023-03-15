from dataclasses import dataclass

@dataclass
class Settings:
  screen_width: int = 1200
  screen_height: int = 800
  bg_color: tuple =(230, 230, 230)
  ship_speed: float = 5