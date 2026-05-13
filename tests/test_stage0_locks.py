"""Stage 0 gate — fail loud if any lock drifts."""
import yaml
from pathlib import Path

CFG = Path(__file__).resolve().parent.parent / "config"
def _load(name): return yaml.safe_load((CFG / name).read_text())


def test_sub_llm_max_turns_is_one():
    assert _load("safeguards.yaml")["rlm_env"]["sub_llm_max_turns"] == 1

def test_sub_max_completion_tokens_is_8192():
    assert _load("safeguards.yaml")["rlm_env"]["sub_max_completion_tokens"] == 8192

def test_sandbox_timeout_minutes_is_one():
    assert _load("safeguards.yaml")["sandbox"]["sandbox_timeout_minutes"] == 1

def test_sandbox_memory_gb_is_four():
    assert _load("safeguards.yaml")["sandbox"]["sandbox_memory_gb"] == 4

def test_code_execution_timeout_is_120():
    assert _load("safeguards.yaml")["sandbox"]["code_execution_timeout"] == 120

def test_abort_on_code_timeout_is_true():
    assert _load("safeguards.yaml")["sandbox"]["abort_on_code_timeout"] is True

def test_deterministic_weight_is_one():
    assert _load("safeguards.yaml")["rubric"]["deterministic_weight"] == 1.0

def test_llm_sim_weight_is_zero_seven():
    assert _load("safeguards.yaml")["rubric"]["llm_sim_weight"] == 0.7

def test_freeform_weight_is_zero_five():
    # updated for geometric coupling (anti-length-grift): freeform_weight is now a LEGACY
    # field — free-form scoring uses (ROUGE_Lsum/100)^0.6 * BERT_F1^0.4 wired at weight=1.0
    # in CurieRubric. The 0.5 value is retained for historical reference (CLAUDE.md guard #7).
    assert _load("safeguards.yaml")["rubric"]["freeform_weight"] == 0.5

def test_judge_family_is_not_alibaba():
    assert _load("safeguards.yaml")["rubric"]["judge_model_family"].lower() != "alibaba"

def test_judge_temperature_is_zero():
    assert _load("judge.yaml")["temperature"] == 0.0

def test_curie_task_count_is_ten():
    t = _load("curie_tasks.yaml")
    assert t["task_count"] == 10
    assert len(t["tasks"]) == 10
    assert len({task["id"] for task in t["tasks"]}) == 10

def test_curie_problem_count_is_578():
    assert _load("curie_tasks.yaml")["problem_count"] == 578

def test_curie_tasks_contain_mpve():
    ids = {t["id"] for t in _load("curie_tasks.yaml")["tasks"]}
    assert "MPVE" in ids
    assert "MPV" not in ids

def test_curie_tasks_contain_qecc_65():
    ids = {t["id"] for t in _load("curie_tasks.yaml")["tasks"]}
    assert "QECC_65" in ids
    assert "QECC" not in ids

def test_dispatcher_retrieval_contains_mpve():
    ids = _load("rubric_dispatcher.yaml")["retrieval_tasks"]["ids"]
    assert "MPVE" in ids
    assert "MPV" not in ids

def test_dispatcher_freeform_contains_qecc_65():
    ids = _load("rubric_dispatcher.yaml")["freeform_tasks"]["ids"]
    assert "QECC_65" in ids
    assert "QECC" not in ids

def test_dispatcher_geometric_weight_matches_safeguards_deterministic():
    s = _load("safeguards.yaml"); d = _load("rubric_dispatcher.yaml")
    assert d["geometric_tasks"]["weight"] == s["rubric"]["deterministic_weight"]

def test_dispatcher_freeform_metric_is_geometric():
    # updated for geometric coupling (anti-length-grift): free-form is one metric, not two.
    d = _load("rubric_dispatcher.yaml")
    assert d["freeform_tasks"]["metric"] == "freeform_geometric"

def test_dispatcher_freeform_formula_lock():
    # updated for geometric coupling (anti-length-grift): lock the exact formula string.
    d = _load("rubric_dispatcher.yaml")
    assert d["freeform_tasks"]["formula"] == "(ROUGE_Lsum/100)^0.6 * BERT_F1^0.4"

def test_dispatcher_freeform_weight_matches_safeguards_deterministic():
    # updated for geometric coupling (anti-length-grift): free-form wired at weight=1.0,
    # parity with programmatic tasks (IoU, ID_r). freeform_weight in safeguards.yaml is LEGACY.
    s = _load("safeguards.yaml"); d = _load("rubric_dispatcher.yaml")
    assert d["freeform_tasks"]["weight"] == s["rubric"]["deterministic_weight"]

def test_dft_c_in_freeform():
    # Stage 3b: DFT-C moved retrieval→freeform per Curie release
    d = _load("rubric_dispatcher.yaml")
    assert "DFT-C" in d["freeform_tasks"]["ids"]
    assert "DFT-C" not in d["retrieval_tasks"]["ids"]

def test_dispatcher_retrieval_weight_matches_safeguards_llmsim():
    s = _load("safeguards.yaml"); d = _load("rubric_dispatcher.yaml")
    assert d["retrieval_tasks"]["weight"] == s["rubric"]["llm_sim_weight"]

def test_frozen_prompts_subdirs_exist():
    assert (CFG / "frozen_prompts" / "llmsim").is_dir()
    assert (CFG / "frozen_prompts" / "lm_score").is_dir()
