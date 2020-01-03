import numpy as np
from scipy import signal
import librosa

_mel_basis = None

def get_hop_size(hparams):
    hop_size = hparams.data.hop_size
    if hop_size is None:
        assert hparams.data.frame_shift_ms is not None
        hop_size = int(hparams.data.frame_shift_ms / 1000 * hparams.data.sr)
    return hop_size

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams.data.nfft, hop_length=get_hop_size(hparams), win_length=hparams.data.win_size)

def _build_mel_basis(hparams):
    assert hparams.data.fmax <= hparams.data.sr // 2
    return librosa.filters.mel(hparams.data.sr , hparams.data.nfft, n_mels=hparams.data.nmels, fmin=hparams.data.fmin, fmax=hparams.data.fmax)  # fmin=0, fmax= sample_rate/2.0

def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)

def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.data.min_level_db / 20 * np.log(10))  # min_level_db = -100
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S, hparams):
    if hparams.data.allow_clipping_in_normalization:
        if hparams.data.symmetric_mels:
            return np.clip((2 * hparams.data.max_abs_value) * (
                        (S - hparams.data.min_level_db) / (-hparams.data.min_level_db)) - hparams.data.max_abs_value,
                           -hparams.data.max_abs_value, hparams.data.max_abs_value)
        else:
            return np.clip(hparams.data.max_abs_value * ((S - hparams.data.min_level_db) / (-hparams.data.min_level_db)), 0,
                           hparams.data.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.data.min_level_db >= 0
    if hparams.data.symmetric_mels:
        return (2 * hparams.data.max_abs_value) * (
                    (S - hparams.data.min_level_db) / (-hparams.data.min_level_db)) - hparams.data.max_abs_value
    else:
        return hparams.data.max_abs_value * ((S - hparams.data.min_level_db) / (-hparams.data.min_level_db))

def melspectrogram(wav, hparams):
    D = _stft(preemphasis(wav, hparams.data.preemphasis, hparams.data.preemphasize), hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D)**hparams.data.power, hparams), hparams) - hparams.data.ref_level_db

    if hparams.data.signal_normalization:
        return _normalize(S, hparams)
    return S