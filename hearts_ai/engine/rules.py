from dataclasses import dataclass


@dataclass(frozen=True)
class HeartsRules:
    """
    Args:
        moon_shot: Determines whether shooting the moon is enabled
        passing_cards: Determines whether passing cards is enabled
    """
    moon_shot: bool = True
    passing_cards: bool = True
