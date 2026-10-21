"""Time IBM's Relay-BP per shot on the gross code, setup excluded (one core)."""
import sys, time
from pathlib import Path
import numpy as np, stim
from relay_bp.stim import SinterDecoder_RelayBP
sys.path.insert(0, str(Path(__file__).parent))
from m0_reference import PARAMS

repo = Path(sys.argv[1]); n = int(sys.argv[2])
name = ("circuit=bicycle_bivariate_144_12_12_memory_Z,distance=12,rounds=12,error_rate=0.003,"
        "noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim")
c = stim.Circuit.from_file(repo / "tests/testdata/bicycle_bivariate" / name)
t0 = time.time()
dec = SinterDecoder_RelayBP(**PARAMS).compile_decoder_for_dem(dem=c.detector_error_model())
t1 = time.time()
det, obs = c.compile_detector_sampler(seed=1).sample(n, separate_observables=True)
packed = np.packbits(det, axis=1, bitorder="little")
t2 = time.time()
pred = dec.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
t3 = time.time()
pred = np.unpackbits(pred, axis=1, bitorder="little")[:, :obs.shape[1]].astype(bool)
print(f"setup {t1-t0:.1f} s; decode {1e3*(t3-t2)/n:.1f} ms/shot; errors {int(np.any(pred != obs, axis=1).sum())}/{n}")
