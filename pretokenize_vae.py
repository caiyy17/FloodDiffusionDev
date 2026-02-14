import os

import numpy as np
import torch
from lightning import seed_everything
from lightning.pytorch.utilities import rank_zero_info
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from utils.initialize import instantiate, load_config

RAW_DATA_PATH = "raw_data"
PATH_FILES = [
    [
        f"{RAW_DATA_PATH}/BABEL_streamed/train_processed.txt",
        f"{RAW_DATA_PATH}/BABEL_streamed/val_processed.txt",
    ],
    [
        f"{RAW_DATA_PATH}/HumanML3D/train.txt",
        f"{RAW_DATA_PATH}/HumanML3D/val.txt",
        f"{RAW_DATA_PATH}/HumanML3D/test.txt",
    ],
]
FEATURE_PATH = [
    "motions",
    "new_joint_vecs",
]
TOKEN_PATH = [
    "TOKENS_20260113_233726_vae_wan_z4_2250000",
    "TOKENS_20260113_233726_vae_wan_z4_2250000",
]
RECOVERED_PATH = [
    "motions_recovered_TOKENS_20260113_233726_vae_wan_z4_2250000",
    "new_joint_vecs_recovered_TOKENS_20260113_233726_vae_wan_z4_2250000",
]


def main():
    # init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    seed_everything(cfg.seed)
    # vae
    vae_model = instantiate(
        cfg.model.target,
        cfg=None,
        hfstyle=False,
        **cfg.model.params,
    )
    vae_ckpt = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae_model.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(
            vae_model.parameters(), decay=cfg.model.ema_decay
        )
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae_model.parameters())
        rank_zero_info(f"Loaded VAE model from {cfg.test_ckpt} with EMA")
    else:
        vae_model.load_state_dict(vae_ckpt["state_dict"], strict=True)
        rank_zero_info(f"Loaded VAE model from {cfg.test_ckpt} w/o EMA")
    vae_model = vae_model.to(device)

    for path_files, feature_path, token_path, recovered_path in zip(
        PATH_FILES, FEATURE_PATH, TOKEN_PATH, RECOVERED_PATH
    ):
        tokenize_motion(
            vae_model, path_files, device, cfg, feature_path, token_path, recovered_path
        )


@torch.no_grad()
def tokenize_motion(
    vae_model, path_files, device, cfg, feature_path, token_path, recovered_path=None
):
    for path_file in path_files:
        print(
            f"Processing {path_file} with feature path: {feature_path}, token path: {token_path}, recovered path: {recovered_path}"
        )
        if os.path.exists(path_file):
            data_path = os.path.dirname(path_file)
            cur_feature_path = os.path.join(data_path, f"{feature_path}")
            cur_token_path = os.path.join(data_path, f"{token_path}")
            if not os.path.exists(cur_token_path):
                os.makedirs(cur_token_path)
            cur_recovered_path = None
            if recovered_path is not None:
                cur_recovered_path = os.path.join(data_path, f"{recovered_path}")
                if not os.path.exists(cur_recovered_path):
                    os.makedirs(cur_recovered_path)
            with open(path_file, "r") as f:
                names = [line.strip() for line in f if line.strip()]
                for name in tqdm(
                    names, desc=f"Processing {os.path.basename(path_file)}"
                ):
                    single_feature_path = f"{cur_feature_path}/{name}.npy"
                    single_token_path = f"{cur_token_path}/{name}.npy"
                    if cur_recovered_path is not None:
                        single_recovered_path = f"{cur_recovered_path}/{name}.npy"
                    if os.path.exists(single_feature_path):
                        try:
                            feature = np.load(single_feature_path, allow_pickle=True)
                            feature = torch.from_numpy(feature).to(device)[None, :, :]
                            token = vae_model.encode(feature)
                            recovered = vae_model.decode(token)
                            np.save(single_token_path, token.detach().cpu().numpy()[0])
                            if cur_recovered_path is not None:
                                np.save(
                                    single_recovered_path,
                                    recovered.detach().cpu().numpy()[0],
                                )
                        except Exception as e:
                            rank_zero_info(f"Error processing {name}: {e}")

    # Compute and save mean and std across all tokens
    data_path = os.path.dirname(path_files[0])
    cur_token_path = os.path.join(data_path, token_path)
    all_tokens = []
    skipped = 0
    for f in tqdm(os.listdir(cur_token_path), desc="Loading tokens for mean/std"):
        if f.endswith(".npy"):
            token = np.load(os.path.join(cur_token_path, f))
            if np.any(np.isnan(token)) or np.any(np.isinf(token)):
                skipped += 1
                rank_zero_info(f"Skipping {f} due to NaN/Inf")
                continue
            all_tokens.append(token)
    if skipped > 0:
        rank_zero_info(f"Skipped {skipped} files with NaN/Inf")

    if all_tokens:
        all_tokens = np.concatenate(all_tokens, axis=0)
        mean = all_tokens.mean(axis=0)
        std = all_tokens.std(axis=0)
        mean_path = os.path.join(data_path, f"Mean_{token_path}.npy")
        std_path = os.path.join(data_path, f"Std_{token_path}.npy")
        np.save(mean_path, mean)
        np.save(std_path, std)
        rank_zero_info(f"Saved mean to {mean_path}, shape: {mean.shape}")
        rank_zero_info(f"Saved std to {std_path}, shape: {std.shape}")


if __name__ == "__main__":
    main()
