import numpy as np
import torch

from DeepSpeaker.deep_speaker.audio import read_mfcc
from DeepSpeaker.deep_speaker.batcher import sample_from_mfcc
from DeepSpeaker.deep_speaker.constants import NUM_FRAMES, SAMPLE_RATE
from DeepSpeaker.deep_speaker.conv_models import DeepSpeakerModel

# Load the Deep Speaker model
deep_speaker_model = DeepSpeakerModel()
deep_speaker_model.m.load_weights(
    "checkpoints/ResCNN_triplet_training_checkpoint_265.h5", by_name=True
)


def extract_speaker_embedding(
    wav_file, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Extract speaker embedding from a WAV file using Deep Speaker."""
    mfcc = sample_from_mfcc(read_mfcc(wav_file, SAMPLE_RATE), NUM_FRAMES)
    embedding = deep_speaker_model.m.predict(
        np.expand_dims(mfcc, axis=0)
    )  # Shape: (1, 512)
    embedding = torch.tensor(embedding, dtype=torch.float32).to(
        device
    )  # Move to correct device
    return embedding  # Shape: (1, 512)
