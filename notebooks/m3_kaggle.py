# syndrome M3a on Kaggle: latency per shot, each shot on the whole GPU, against IBM's decoder.
# Notebook settings: Accelerator "GPU T4 x2" (or x1), Internet on. Paste this whole file into one
# cell and run it. About 25 minutes; most of it is IBM's decoder on one CPU thread (the reference
# for every shot). Then download syndrome_m3_results.zip from the output panel.
import glob
import os
import subprocess
import sys
import zipfile

W = "/kaggle/working"
os.chdir(W)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


sh("rm -rf syndrome relay && git clone -q https://github.com/Shakhtar-Sankur/syndrome "
   "&& git clone -q https://github.com/trmue/relay && git -C relay checkout -q d185194")
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")

# IBM's decoder is Rust with Python bindings: install a Rust toolchain, then the package.
sh("curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal")
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":/usr/local/cuda/bin:" + os.environ["PATH"]
# Same versions as the M0 and M1 runs; relay's [stim] extra brings beliefmatching, which
# relay_bp.stim uses to turn a circuit into check matrices.
sh(f"{sys.executable} -m pip install -q stim==1.16.0 sinter==1.16.0 beliefmatching==0.2.0 './relay[stim]'")

# The CUDA kernel, built for the GPU in this session (sm_75 on a T4, sm_60 on a P100).
# -fmad=false keeps every rounding step the same as IBM's code, in float64 and float32.
cc = subprocess.run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader", shell=True,
                    capture_output=True, text=True).stdout.split()[0].replace(".", "")
sh(f"cd syndrome/csrc && nvcc -O3 -fmad=false -std=c++17 -arch=sm_{cc} -Xcompiler -fPIC -shared "
   "-o libsyndrome_cuda.so relay_bp_cuda.cu")

sh(f"set -o pipefail; {sys.executable} syndrome/bench/m3_latency.py --relay-repo relay "
   f"--out {W}/results/m3_t4_latency.json --shots 64 --sweep-shots 16 --dtypes f32,f64 "
   f"2>&1 | tee {W}/results_m3.log")

with zipfile.ZipFile(f"{W}/syndrome_m3_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{W}/results/*") + [f"{W}/results_m3.log"]:
        if os.path.exists(f):
            z.write(f, os.path.relpath(f, W))
print("Download syndrome_m3_results.zip from the output panel.")
