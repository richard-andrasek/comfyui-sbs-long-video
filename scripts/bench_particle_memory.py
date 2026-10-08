"""Measure particle processing time and peak process/device memory."""
import argparse
import os
import time

import torch

from particle_depth import apply_particle_depth, detect_particle_mask


def peak_rss_mb():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 if os.name == "nt" else 1024 * 1024)
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return counters.PeakWorkingSetSize / (1024 * 1024)
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sizes', nargs='+', default=['1080p', '4k'])
    parser.add_argument('--chunks', nargs='+', type=int, default=[8, 16, 32])
    args = parser.parse_args()
    device = torch.device(args.device)
    sizes = {'1080p': (1080, 1920), '4k': (2160, 3840)}
    print('size,chunk,device,ms_per_frame,peak_rss_mb,peak_cuda_mb')
    for size in args.sizes:
        h, w = sizes[size]
        for batch in args.chunks:
            frames = torch.rand((batch, h, w, 3), device=device)
            frames[frames < 0.995] *= 0.2
            depth = torch.rand((batch, 1, h, w), device=device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            mask = detect_particle_mask(frames)
            apply_particle_depth(depth, mask, return_synthetic=False)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                cuda_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            else:
                cuda_mb = 0
            elapsed = time.perf_counter() - started
            print(f'{size},{batch},{device},{elapsed * 1000 / batch:.2f},{peak_rss_mb():.1f},{cuda_mb:.1f}', flush=True)
            del frames, depth, mask


if __name__ == '__main__':
    main()
