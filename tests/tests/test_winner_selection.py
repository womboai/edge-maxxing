import json
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from unittest import TestCase

from base.checkpoint import Uid, Key
from weight_setting.winner_selection import get_contestant_ranks, calculate_rank_weights

TEST_DATA_DIRECTORY = Path(__file__).parent.parent / "test_data"

class WinnerSelectionTest(TestCase):
    def test_winner_selection(self):

        for path in TEST_DATA_DIRECTORY.glob("*.json"):
            winners: dict[Key, list[Uid]] = defaultdict(list)
            date: datetime = datetime.strptime(path.stem, "%Y-%m-%d")
            with path.open("r") as file:
                for uid, data in json.load(file).items():
                    scores = {key: info["score"] for key, info in data.items()}
                    submitted_blocks = {key: info["block"] for key, info in data.items()}

                    ranks = get_contestant_ranks(scores)
                    weights = calculate_rank_weights(submitted_blocks, ranks)

                    winner = max(
                        weights.items(),
                        key=itemgetter(1),
                        default=None
                    )

                    if not winner:
                        continue

                    winners[winner[0]].append(uid)

            msg = f"Multiple winners found on {date.strftime('%Y-%m-%d')}:\n" + "\n".join(
                f"{winner}: {', '.join(map(str, validator_uids))}"
                for winner, validator_uids in winners.items()
            )

            self.assertEqual(len(winners), 1, msg=msg)
