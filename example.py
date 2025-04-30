import numpy as np
from bungee_python import bungee

sample_rate = 44100
channels = 1
duration_seconds = 5
frequency = 440


t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds))
input_audio = 0.5 * np.sin(2. * np.pi * frequency * t)
input_audio = input_audio.astype(np.float32)
if channels == 1:
    input_audio = input_audio[:, np.newaxis]
elif channels == 2:
    input_audio = np.stack([input_audio, input_audio], axis=-1)

print(f"Input shape: {input_audio.shape}")


stretcher = bungee.Bungee(sample_rate=sample_rate, channels=channels)


stretcher.set_speed(0.5)
stretcher.set_pitch(1.0)


output_audio = stretcher.process(input_audio)

print(f"Output shape: {output_audio.shape}")