import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa


# Modules for harmonic filters
def hz_to_midi(hz):
    return 12 * (torch.log2(hz) - np.log2(440.0)) + 69


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def note_to_midi(note):
    return librosa.core.note_to_midi(note)


def hz_to_note(hz):
    return librosa.core.hz_to_note(hz)


def initialize_filterbank(sample_rate, n_harmonic, semitone_scale):
    # MIDI
    # lowest note
    low_midi = note_to_midi("C1")

    # highest note
    high_note = hz_to_note(sample_rate / (2 * n_harmonic))
    high_midi = note_to_midi(high_note)

    # number of scales
    level = (high_midi - low_midi) * semitone_scale
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])

    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i + 1)))

    return harmonic_hz, level


class HarmonicSTFT(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        win_length=None,
        hop_length=None,
        pad=0,
        power=1,
        normalized=False,
        n_harmonic=6,
        semitone_scale=2,
        bw_Q=1.0,
        learn_bw=None,
    ):
        super(HarmonicSTFT, self).__init__()

        # Parameters
        self.sample_rate = sample_rate
        self.n_harmonic = n_harmonic
        self.bw_alpha = 0.1079
        self.bw_beta = 24.7

        # Spectrogram
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=None,
            pad=0,
            window_fn=torch.hann_window,
            power=power,
            normalized=normalized,
            wkwargs=None,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize the filterbank. Equally spaced in MIDI scale.
        harmonic_hz, self.level = initialize_filterbank(
            sample_rate, n_harmonic, semitone_scale
        )

        # Center frequncies to tensor
        self.f0 = torch.tensor(harmonic_hz.astype("float32"))

        # Bandwidth parameters
        if learn_bw == "only_Q":
            self.bw_Q = nn.Parameter(
                torch.tensor(np.array([bw_Q]).astype("float32"))
            )
        elif learn_bw == "fix":
            self.bw_Q = torch.tensor(np.array([bw_Q]).astype("float32"))

    def get_harmonic_fb(self):
        # bandwidth
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = bw.unsqueeze(0)  # (1, n_band)
        f0 = self.f0.unsqueeze(0)  # (1, n_band)
        fft_bins = self.fft_bins.unsqueeze(1)  # (n_bins, 1)

        up_slope = torch.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        down_slope = torch.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        fb = torch.max(self.zero, torch.min(down_slope, up_slope))
        return fb

    def to_device(self, device, n_bins):
        self.f0 = self.f0.to(device)
        self.bw_Q = self.bw_Q.to(device)
        # fft bins
        self.fft_bins = torch.linspace(0, self.sample_rate // 2, n_bins)
        self.fft_bins = self.fft_bins.to(device)
        self.zero = torch.zeros(1)
        self.zero = self.zero.to(device)

    def forward(self, waveform):
        # stft
        spectrogram = self.spec(waveform)

        # to device
        self.to_device(waveform.device, spectrogram.size(1))

        # triangle filter
        harmonic_fb = self.get_harmonic_fb()
        harmonic_spec = torch.matmul(
            spectrogram.transpose(1, 2), harmonic_fb
        ).transpose(1, 2)

        # (batch, channel, length) -> (batch, harmonic, f0, length)
        b, c, time = harmonic_spec.size()
        harmonic_spec = harmonic_spec.view(
            b,
            self.n_harmonic,
            self.level,
            time
        )

        # amplitude to db
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        return harmonic_spec
