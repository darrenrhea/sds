import numpy as np
from pathlib import Path


def check_todo_task(todo_task):
    """
    Makee sure you know what you are enqueueing into seg_queue_todo,
    the todo queue.
    """
    if todo_task is not None:
        assert len(todo_task) == 2
        assert isinstance(todo_task[0], Path)
        assert isinstance(todo_task[1], Path)
