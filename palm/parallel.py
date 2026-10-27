import multiprocessing as mp
import numpy as np

import os, pickle, hashlib
CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "cache")


def _key(args):
    return hashlib.md5(pickle.dumps(args, protocol=4)).hexdigest()


def _job(args):
    """Run one (params, scheme, frames, seed, kwargs) job; results are cached on disk
    so that an interrupted experiment resumes where it stopped."""
    path = os.path.join(CACHE, _key(args) + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    from palm.solver import warmup
    from palm.agents import run
    warmup()
    p, scheme, frames, seed, kw = args
    o = run(p, scheme, frames=frames, seed=seed, **kw)
    os.makedirs(CACHE, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(o, f)
    os.replace(path + ".tmp", path)
    return o

def pmap(jobs, procs=2):
    """jobs: list of (p, scheme, frames, seed, kwargs)"""
    # 'fork' on Linux/Jetson; Windows/macOS have no fork -> 'spawn'
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(procs) as pool:
        return pool.map(_job, jobs, chunksize=1)

def summarize(o, tail=0.3):
    n = len(o["Q"]); s = slice(int(n * (1 - tail)), None)
    return dict(Q=o["Q"][s].mean(), E=o["E"][s].mean(), wrate=o["wrate"][s].mean(),
                off=o["off"][s].mean(), Ees=o["Ees"][s].mean(), hs=o["hs"][s].mean(),
                hpro=o["hpro"][s].mean() if "hpro" in o else 0.0,
                dt=np.median(o["dt"]), Qslope=np.polyfit(np.arange(n)[s], o["Q"][s], 1)[0])
