"""Determinism audit: the keyed constructions must be bit-identical across processes, runs and
hash seeds, or two ends on different machines could not build the same aperture (the whole
'nothing pre-shared but the key' claim). Each worker process prints sha256 of the key kernels;
the parent runs it twice with different PYTHONHASHSEED and compares.
"""
import hashlib, json, os, subprocess, sys
import numpy as np

def digest():
    from entroptics_rf import waveform as wf
    out = {}
    def h(name, a):
        out[name] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    for key in (1234, None, "alice", b"\x07\x08"):
        k = str(key)
        h(f"basis/{k}", wf.mode_basis(64, 64, key))
        h(f"modekeys/{k}", wf.mode_keys(64, key, None, first=0, rows=40))
        h(f"radius/{k}", wf.row_radius(64, key, first=0, rows=40) if hasattr(wf, "row_radius") else np.zeros(1))
        h(f"ref/{k}", wf._reference_row(64, key))
    bits = np.random.default_rng(0).integers(0, 2, 64*2*30)
    h("compose/1234", wf.compose(bits, 64, 64, 1234))
    h("compose/None", wf.compose(bits, 64, 64, None))
    h("compose_qam/16", wf.compose_qam(np.random.default_rng(1).integers(0,2,64*4*20), 64, 64, 1234, M=16))
    return out

if __name__ == "__main__":
    if "--worker" in sys.argv:
        print(json.dumps(digest())); sys.exit(0)
    runs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH="src")
        r = subprocess.run([sys.executable, __file__, "--worker"], capture_output=True, text=True, env=env)
        runs.append(json.loads(r.stdout))
    # also a round-trip correctness check: decode(compose) is bit-exact
    sys.path.insert(0, "src"); from entroptics_rf import waveform as wf
    bits = np.random.default_rng(0).integers(0, 2, 64*2*120)   # a realistic burst (blind width needs >~80 rows)
    got, d = wf.decode(np.concatenate([np.zeros(37, complex), wf.compose(bits, 64, 64, 1234)]), 1234)
    want = bits[2*64*d.row0:]; n = min(got.size, want.size)
    rt = bool(n) and np.array_equal(got[:n], want[:n])
    # seed=None (open mode) draws a sender-random per-row radius by design, so radius/None and
    # compose/None SHOULD differ run to run; exclude them from the 'must be identical' set.
    keyed = lambda k: not k.endswith('/None') or k.startswith(('basis/', 'modekeys/', 'ref/'))
    diffs = [k for k in runs[0] if keyed(k) and runs[0][k] != runs[1].get(k)]
    same = not diffs
    print(f"keyed kernels bit-identical across PYTHONHASHSEED 0 vs 12345: {same}")
    if diffs: print("  DIFFER:", diffs)
    print(f"compose->decode round trip bit-exact: {rt}")
    print("digests:", json.dumps(runs[0], indent=0)[:400])
    sys.exit(0 if same and rt else 1)
