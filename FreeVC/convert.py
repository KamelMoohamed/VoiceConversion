import argparse
import logging
import os
import time

import librosa
import torch
from scipy.io.wavfile import write
from tqdm import tqdm

import utils
from mel_processing import mel_spectrogram_torch
from models import SynthesizerTrn, extract_speaker_embedding  # Using Deep Speaker
from wavlm import WavLM, WavLMConfig

logging.getLogger("numba").setLevel(logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpfile",
        type=str,
        default="configs/freevc.json",
        help="path to json config file",
    )
    parser.add_argument(
        "--ptfile", type=str, default="checkpoints/freevc.pth", help="path to pth file"
    )
    parser.add_argument(
        "--txtpath", type=str, default="convert.txt", help="path to txt file"
    )
    parser.add_argument(
        "--outdir", type=str, default="output/freevc", help="path to output dir"
    )
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = utils.get_cmodel(0)

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

            g_tgt = extract_speaker_embedding(tgt).to(device)
            g_tgt = g_tgt.unsqueeze(0).unsqueeze(-1)  # Match expected shape

            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            wav_src = torch.from_numpy(wav_src).to(device).unsqueeze(0)
            c = utils.get_content(cmodel, wav_src).to(device)

            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(device)

            audio = net_g.infer(
                c.to(device), wav_file=tgt, c_lengths=c_lengths
            )  # Pass everything to CUDA

            audio = audio[0][0].data.cpu().float().numpy()

            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(
                    os.path.join(args.outdir, f"{timestamp}_{title}.wav"),
                    hps.data.sampling_rate,
                    audio,
                )
            else:
                write(
                    os.path.join(args.outdir, f"{title}.wav"),
                    hps.data.sampling_rate,
                    audio,
                )
