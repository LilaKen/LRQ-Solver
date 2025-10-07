import paddle.profiler as profiler


def init_profiler(enable_profiler):
    if enable_profiler:
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            timer_only=False,
            scheduler=(3, 7),
            on_trace_ready=profiler.export_chrome_tracing("./log"),
        )
        prof.start()
        return prof
    else:
        return None


def update_profiler(enable_profiler, prof, ep):
    if enable_profiler:
        prof.step()
        if ep == 10:
            prof.stop()
            prof.summary(
                sorted_by=profiler.SortedKeys.GPUTotal,
                op_detail=True,
                thread_sep=False,
                time_unit="ms",
            )
            exit()
    return prof
