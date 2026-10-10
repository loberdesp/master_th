from master_th.audio import gain_to_db, load, rms


def main() -> None:
    x, sr = load("data/example_audio_file.wav")
    print(f"shape={x.shape} dtype={x.dtype} sr={sr} rms={float(gain_to_db(rms(x))):.1f} dBFS")
