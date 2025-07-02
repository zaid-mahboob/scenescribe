import noisereduce as nr
import scipy.io.wavfile as wavfile
import numpy as np


def denoise_wav(file_path):
    """
    Loads a WAV file, performs noise reduction, and overwrites the original file.
    Uses spectral gating (via noisereduce).
    """
    print(f"🔧 Denoising audio: {file_path}")

    # Load audio
    rate, data = wavfile.read(file_path)

    # If stereo, convert to mono
    if len(data.shape) == 2:
        data = np.mean(data, axis=1).astype(np.int16)

    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    # Convert back to int16 if needed
    if reduced_noise.dtype != np.int16:
        reduced_noise = np.clip(reduced_noise, -32768, 32767).astype(np.int16)

    # Overwrite the original file
    wavfile.write(file_path, rate, reduced_noise)

    print("✅ Noise reduction complete and file saved.")


# Example usage:
denoise_wav("recorded_audio.wav")
