"""Rule-based opponent for Leduc Hold'em."""

import numpy as np


class RuleBasedOpponent:
    """
    Rule-based opponent with simple card-strength strategy.

    Strategy:
    - K (strongest, index 2): Bet or Raise
    - Q (medium, index 1): Check or Call
    - J (weakest, index 0): Check or Fold vs aggression
    """

    def __init__(self):
        """Initialize rule-based opponent."""
        pass

    def get_action(self, observation, legal_actions):
        """
        Select action based on hand strength.

        Args:
            observation: 36-dimensional observation vector
            legal_actions: Boolean mask of legal actions

        Returns:
            Selected action index
        """
        # Extract hand card (indices 0-2 are one-hot encoded: J, Q, K)
        hand = observation[:3]
        card_index = np.argmax(hand)  # 0=J, 1=Q, 2=K

        # Action indices: 0=Call, 1=Raise, 2=Fold, 3=Check

        if card_index == 2:  # K (strongest)
            # Prefer Raise, fallback to Call or Check
            if legal_actions[1]:  # Raise
                return 1
            elif legal_actions[0]:  # Call
                return 0
            elif legal_actions[3]:  # Check
                return 3
            else:  # Fold (last resort)
                return 2

        elif card_index == 1:  # Q (medium)
            # Prefer Check or Call
            if legal_actions[3]:  # Check
                return 3
            elif legal_actions[0]:  # Call
                return 0
            elif legal_actions[1]:  # Raise
                return 1
            else:  # Fold
                return 2

        else:  # J (weakest, card_index == 0)
            # Prefer Check or Fold
            if legal_actions[3]:  # Check
                return 3
            elif legal_actions[2]:  # Fold
                return 2
            elif legal_actions[0]:  # Call
                return 0
            else:  # Raise (last resort)
                return 1
