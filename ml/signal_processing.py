# coding=utf-8

from pydub import AudioSegment
from . import system


def load_wav(filename):
    return AudioSegment.from_wav(filename)


def wav_duration(filename):
    # in seconds
    dur = system.run_command("soxi -D {}".format(filename), verbose=False)
    dur.strip()
    return float(dur)


def sliding_window(sequence, window_size_in_frames, step_in_frames, time_limits, t0, freq, debugging=False):
    frame_duration = 1.0 / freq
    sec_length = sequence.shape[1]
    num_of_chunks = ((sec_length - window_size_in_frames) // step_in_frames) + 1
    ti, tf = time_limits

    for i in range(0, num_of_chunks * step_in_frames, step_in_frames):
        frame_initial_time, frame_end_time = frame_time_limits(i, freq, t0, window_size_in_frames)

        if frame_initial_time < ti or frame_end_time > tf:
            continue
        if debugging:
            step_in_secs = step_in_frames * frame_duration
            print(("frame index:", i, "from:", frame_initial_time, "to:", frame_end_time, "next should be:", frame_initial_time + step_in_secs, "step:", step_in_secs))
        yield i, sequence[:, i:i + window_size_in_frames]


def time_sliding_window(sequence, window_size_in_secs, step_in_secs, time_limits, t0, freq, debugging=False):
    frame_duration = 1.0 / freq
    window_size_in_frames = int(window_size_in_secs / frame_duration)  # for example 100 ms = 51 frames aprox if freq = 512.0
    step_in_frames = int(step_in_secs / frame_duration)
    return sliding_window(sequence, window_size_in_frames, step_in_frames, time_limits, t0, freq, debugging)


def interpolate(x, y, interpolator, deg):
    return interpolator(x, y, deg=deg)


def frame_time_limits(sample_number, freq, t0, window_size_in_frames):
    frame_duration = 1.0 / freq
    window_size_in_secs = window_size_in_frames * frame_duration
    frame_initial_time = t0 + frame_duration * sample_number
    frame_end_time = frame_initial_time + window_size_in_secs
    return frame_initial_time, frame_end_time


def frame_at(t, freq, tmin):
    relative_time = t - tmin
    return int(relative_time * freq)


def extract_mfccss(filename):
    # run "SMILExtract -C config/MFCC12_E_D_A.conf -I input.wav -O output.mfcc.htk"
    pass
