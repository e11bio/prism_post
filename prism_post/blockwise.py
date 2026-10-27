"""
Small helper around volara's run_blockwise.

run_blockwise returns False if any block failed rather than raising, which is
easy to miss in a chain of scripts. This raises instead, so a failed step
stops you right there. Worker logs with the actual errors are in
<data_dir>/volara_logs/<task name>/.
"""

from volara.blockwise import BlockwiseTask


def run(task: BlockwiseTask, multiprocessing: bool = True) -> None:
    ok = task.run_blockwise(multiprocessing=multiprocessing)
    if not ok:
        raise RuntimeError(
            f"{task.task_name} had failed blocks, see {task.meta_dir.parent / task.task_name}"
        )
