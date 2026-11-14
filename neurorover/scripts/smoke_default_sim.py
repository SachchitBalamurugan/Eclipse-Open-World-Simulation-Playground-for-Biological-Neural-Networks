"""Verification 1: closed loop against the default CL simulator (no custom culture)."""
import time
import cl

def main():
    with cl.open() as neurons:
        print("channels", neurons.get_channel_count(), "fps", neurons.get_frames_per_second())
        n_spk = n_stim = 0
        t0 = time.perf_counter()
        for tick in neurons.loop(ticks_per_second=100, stop_after_seconds=2):
            n_spk += len(tick.analysis.spikes)
            n_stim += len(tick.analysis.stims)
            if tick.iteration % 10 == 0:
                neurons.stim(cl.ChannelSet(1), cl.StimDesign(160, -1.0, 160, 1.0))
        print(f"ticks ok; spikes={n_spk} stims_seen={n_stim} wall={time.perf_counter()-t0:.2f}s")

if __name__ == "__main__":
    main()
