import os.path
import re
import shutil
from collections import deque
from typing import Optional

import numpy as np

SAVE_DIR = "game_saves"


class CheckpointHandler:
    """
    Stores checkpoint names together with their "rank" (index of the milestone).
    Allows to sample a checkpoint name uniformly over the ranks.
    """
    def __init__(
            self,
            max_checkpoints: int,
            num_milestones: int = 100,
            initial_checkpoints: Optional[list[tuple[int, str]]] = None
    ):
        self.max_checkpoints = max_checkpoints
        self.checkpoint_names = [deque(maxlen=max_checkpoints) for _ in range(num_milestones)]
        self.max_reached_milestone = 0  # the index of the first milestone is 1
        if initial_checkpoints:
            for milestone_index, checkpoint_name in initial_checkpoints:
                self.checkpoint_names[milestone_index].append(checkpoint_name)
                if milestone_index > self.max_reached_milestone:
                    self.max_reached_milestone = milestone_index

    def add_checkpoint(self, message_string: str):
        assert os.path.exists(SAVE_DIR), f"Directory {SAVE_DIR} does not exist. Currently in {os.getcwd()}"
        try:
            if ((milestone_match := re.search(r"Milestone completed: *(\d+)", message_string))
                    and (save_match := re.search(r"Game saved as: *(.+?)(?:\s*$|\n)", message_string))):
                milestone_index = int(milestone_match.group(1))
                save_name = save_match.group(1)
                if milestone_index > self.max_reached_milestone:
                    self.max_reached_milestone = milestone_index

                checkpoint_deque = self.checkpoint_names[milestone_index]
                if len(checkpoint_deque) >= self.max_checkpoints:
                    outdated_checkpoint = checkpoint_deque[0]  # oldest item
                    if os.path.exists(path := os.path.join(SAVE_DIR, outdated_checkpoint)):
                        shutil.rmtree(path)
                # deque with maxlen automatically removes oldest when full
                checkpoint_deque.append(save_name)
                return {"milestone_index": milestone_index, "save_name": save_name}
        except Exception as e:
            # rmtree may fail if 2 runs use the same directory
            print(f"Error adding checkpoint: {e}")
        return None

    def sample_checkpoint(self) -> str:
        if self.max_reached_milestone < 1:
            return ""
        milestone_index = np.random.randint(1, self.max_reached_milestone + 1)
        return np.random.choice(list(self.checkpoint_names[milestone_index]))
