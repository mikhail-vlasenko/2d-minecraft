import re
from queue import Queue
from typing import Optional

import numpy as np


class CheckpointHandler:
    """
    Stores checkpoint names together with their "rank" (index of the milestone).
    Allows to sample a checkpoint name uniformly over the ranks.
    """
    def __init__(
            self,
            num_milestones: int,
            max_checkpoints: int,
            initial_checkpoints: Optional[list[tuple[int, str]]] = None
    ):
        self.checkpoint_names = [Queue(maxsize=max_checkpoints) for _ in range(num_milestones)]
        self.max_reached_milestone = 0  # the index of the first milestone is 1
        if initial_checkpoints:
            for milestone_index, checkpoint_name in initial_checkpoints:
                self.checkpoint_names[milestone_index].put(checkpoint_name)
                if milestone_index > self.max_reached_milestone:
                    self.max_reached_milestone = milestone_index

    def add_checkpoint(self, message_string: str):
        if ((milestone_match := re.search(r"Milestone completed: *(\d+)", message_string))
                and (save_match := re.search(r"Game saved as: *(.+?)(?:\s*$|\n)", message_string))):
            milestone_index = int(milestone_match.group(1))
            save_name = save_match.group(1)
            if milestone_index > self.max_reached_milestone:
                self.max_reached_milestone = milestone_index
            if self.checkpoint_names[milestone_index].full():
                outdated_checkpoint = self.checkpoint_names[milestone_index].get()
                # todo: delete outdated_checkpoint
            self.checkpoint_names[milestone_index].put(save_name)
            print(f"Added checkpoint {save_name} to milestone {milestone_index}")
            return {"milestone_index": milestone_index, "save_name": save_name}
        return None

    def sample_checkpoint(self) -> str:
        if self.max_reached_milestone < 1:
            return ""
        milestone_index = np.random.randint(1, self.max_reached_milestone + 1)
        # sample a random checkpoint from the milestone
        return np.random.choice(list(self.checkpoint_names[milestone_index].queue))
