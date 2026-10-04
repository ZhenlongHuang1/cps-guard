"""对照方法；保留统一函数导入接口。"""
from .common import BASELINE_COLUMNS
from .nete import import_nete_scores, prepare_nete, run_nete_official
from .onion import onion_deletion_score, run_onion_adapted
from .random import random_baseline
from .rap import prepare_rap_variants, score_rap_responses
