import os
import argparse

from examples.hubert.simple_kmeans import (
    custom_dump_hubert_feature,
    dump_km_label,
)
from examples.speech_matrix.data_helper.model_cfg import (
    kmeans_hub,
)

manifest_key = "s2u_manifests"

def extract_lang_units(
    aud_manifest,
    lang,
    km_save_dir,
    hubert_ckpt,
):
    manifest_dir = os.path.dirname(aud_manifest)
    split = os.path.basename(aud_manifest)
    if split.endswith(".tsv"):
        split = split[:-4]
    feat_dir = os.path.join(km_save_dir, "tmp_features")
    os.makedirs(feat_dir, exist_ok=True)

    # hubert feature
    ckpt_path = hubert_ckpt
    print(hubert_ckpt)
    layer=12
    custom_dump_hubert_feature.main(
        manifest_dir, split, ckpt_path, layer, 1, 0, feat_dir, max_chunk=1600000
    )

    ## kmeans label
    #km_path = os.path.join(hubert_model_dir, kmeans_hub[lang])
    #dump_km_label.dump_label(feat_dir, split, km_path, 1, 0, km_save_dir)

    #cmd = f"mv {km_save_dir}/{split}_0_1.km {km_save_dir}/{split}.km"
    #os.system(cmd)

    ##os.system(f"rm {feat_dir}/{split}*")
    #print(f"done unit extraction: {split}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser("FLEURS shit set preparation")
    parser.add_argument("--proc-fleurs-dir", type=str, required=True)
    parser.add_argument("--save-root", type=str, required=True)
    parser.add_argument("--hubert-ckpt", type=str, required=True)
    args = parser.parse_args()

    manifest_root = os.path.join(args.save_root, manifest_key)
    fleurs_manifest_dir = os.path.join(args.proc_fleurs_dir, "aud_manifests")

    for src_lang_code in ['es']: 
        for tgt_lang_code in ['en']:
            manifest_dir = os.path.join(
                fleurs_manifest_dir,
                f"{src_lang_code}-{tgt_lang_code}",
            )
            src_split = f"shit_{src_lang_code}-{tgt_lang_code}_{src_lang_code}"
            src_aud_manifest = os.path.join(manifest_dir, f"{src_split}.tsv")

            extract_lang_units(
                src_aud_manifest,
                src_lang_code,
                manifest_dir,
                args.hubert_ckpt,
            )

