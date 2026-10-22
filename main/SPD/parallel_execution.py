"""使用 spawn 有界执行独立实验；结果写入与汇总回调保留在父进程。

Windows 入口须放在 ``if __name__ == '__main__'`` 下。取消为合作式：工作进程
忽略 SIGINT，父进程设置共享事件，run_single 在轮次之间检查后退出。
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import signal
import threading


_CANCEL_EVENT = None


class ParallelRunError(RuntimeError):
    """标明失败实验，并保留原异常供调用方记录。"""

    def __init__(self, run_id, original_exception):
        self.run_id = run_id
        self.original_exception = original_exception
        self.cause = original_exception
        super().__init__(f"并行实验 run_id={run_id} 失败：{original_exception}")


def _initialize_worker(cancel_event):
    global _CANCEL_EVENT
    _CANCEL_EVENT = cancel_event
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _execute_run(kwargs):
    # 顶层函数与工作进程内导入支持 Windows spawn，且不依赖 runner 的实验默认值。
    from .run_experiment import run_single

    if _CANCEL_EVENT.is_set():
        raise InterruptedError("并行实验已取消。")
    parameters = dict(kwargs)
    parameters.update(quiet=True, cancel_check=_CANCEL_EVENT.is_set)
    return run_single(**parameters)


def run_parallel(tasks, num_workers, on_result):
    """执行 ``[(run_id, run_single_kwargs), ...]``，成功后回调 ``on_result(id, summary)``。

    最多同时提交 num_workers 个在途实验，完成顺序可以不同于输入顺序。
    失败或 Ctrl+C 后停止提交、通知取消，并在所有工作进程退出后回收已成功结果。
    每个结果的回调最多执行一次；随后抛出 ParallelRunError 或原 KeyboardInterrupt。
    """
    if type(num_workers) is not int or num_workers < 1:
        raise ValueError("num_workers 必须为正整数。")
    if not callable(on_result):
        raise TypeError("on_result 必须为可调用对象。")
    if not tasks:
        return

    context = multiprocessing.get_context("spawn")
    cancel_event = context.Event()
    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(cancel_event,),
    )
    pending = {}
    callback_attempted = set()
    next_task = 0
    active_run_id = None
    failure = None
    try:
        while pending or next_task < len(tasks):
            while len(pending) < num_workers and next_task < len(tasks):
                active_run_id, kwargs = tasks[next_task]
                future = executor.submit(_execute_run, kwargs)
                pending[future] = active_run_id
                next_task += 1
            active_run_id = None
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                active_run_id = pending[future]
                summary = future.result()
                # 回调自身报错/中断时也不能重复调用；其他完成任务仍在 pending 中。
                callback_attempted.add(future)
                on_result(active_run_id, summary)
                del pending[future]
                callback_attempted.discard(future)
            active_run_id = None
    except KeyboardInterrupt as exc:
        failure = exc
    except BaseException as exc:
        failure = ParallelRunError(active_run_id, exc)
    finally:
        # 回收阶段再次 Ctrl+C 不应中断 join，使工作进程遗留在后台。
        old_sigint = None
        if threading.current_thread() is threading.main_thread():
            old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if failure is not None:
                cancel_event.set()
                for future in pending:
                    future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            # 某任务失败/中断时，另一个任务可能刚好完成；保留这些成功产物。
            for future, run_id in pending.items():
                if future.cancelled() or future in callback_attempted:
                    continue
                try:
                    summary = future.result()
                except BaseException:
                    # 包含收到取消事件后退出的 InterruptedError；均不能计为成功。
                    continue
                callback_attempted.add(future)
                try:
                    on_result(run_id, summary)
                except BaseException as exc:
                    if failure is None:
                        failure = exc if isinstance(exc, KeyboardInterrupt) else ParallelRunError(run_id, exc)
        finally:
            if old_sigint is not None:
                signal.signal(signal.SIGINT, old_sigint)

    if isinstance(failure, ParallelRunError):
        raise failure from failure.original_exception
    if failure is not None:
        raise failure
