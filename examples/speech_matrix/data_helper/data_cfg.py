# EuroParl-ST
EP_LANGS = ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ro"]


# VoxPopuli
VP_FAMS = [
    "germanic",
    "uralic",
]
VP_LANGS = [
    "cs",
    "de",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hr",
    "hu",
    "it",
    "lt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
]
VP_LANG_PAIRS = []
for src_lang in VP_LANGS:
    for tgt_lang in VP_LANGS:
        if src_lang == tgt_lang:
            continue
        VP_LANG_PAIRS.append(f"{src_lang}-{tgt_lang}")

high_res_pairs = set(
    [
        "en-es",
        "en-fr",
        "de-en",
        "en-it",
        "en-pl",
        "en-pt",
        "es-fr",
        "es-it",
        "es-pt",
        "fr-it",
        "fr-pt",
        "it-pt",
        "en-nl",
    ]
)
mid_res_pairs = set(
    [
        "en-ro",
        "cs-de",
        "de-es",
        "de-pl",
        "en-sk",
        "cs-fr",
        "es-nl",
        "en-fi",
        "cs-es",
        "en-hu",
        "cs-en",
        "hu-it",
        "nl-pt",
        "pl-pt",
        "nl-pl",
        "de-fr",
        "de-it",
        "fr-nl",
        "it-nl",
        "es-pl",
        "cs-pl",
        "de-nl",
        "es-ro",
        "it-pl",
        "fr-ro",
        "de-pt",
        "pl-sk",
        "cs-it",
        "it-ro",
        "cs-nl",
        "cs-sk",
        "fr-pl",
    ]
)


# FLEURS
FLEURS_LANGS = [
    "cs_cz",
    "de_de",
    "en_us",
    "es_419",
    "et_ee",
    "fi_fi",
    "fr_fr",
    "hr_hr",
    "hu_hu",
    "it_it",
    "lt_lt",
    "nl_nl",
    "pl_pl",
    "pt_br",
    "ro_ro",
    "sk_sk",
    "sl_si",
]
FLORES_LANG_MAP = { # manually fixed language map according to https://huggingface.co/datasets/facebook/flores/blob/main/flores.py
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "it": "ita_Latn",
    "lt": "lit_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
}

DOWNLOAD_HUB = "https://dl.fbaipublicfiles.com/speech_matrix"

manifest_prefix = "train_mined"

# sub directories under save_root/
audio_key = "audios"
aligned_speech_key = "aligned_speech"
manifest_key = "s2u_manifests"
hubert_key = "hubert"
vocoder_key = "vocoder"
s2s_key = "s2s_models"
