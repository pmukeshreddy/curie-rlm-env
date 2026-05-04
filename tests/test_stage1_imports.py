"""Stage 1 gate — verify Prime libraries expose the classes we'll use.
Imports verbatim from verifiers/envs/experimental/rlm_env.py source.

NOTE on prime-rl: per CLAUDE.md L9, prime-rl is cloned + installed separately
on the GPU pod at Stage 5; it is NOT a pyproject dependency, so no
`import prime_rl` test belongs in this local-side gate.
"""

def test_rlm_env_importable():
    from verifiers.envs.experimental.rlm_env import RLMEnv
    assert RLMEnv is not None

def test_rubric_group_top_level():
    import verifiers as vf
    assert vf.RubricGroup is not None

def test_judge_rubric_top_level():
    import verifiers as vf
    assert vf.JudgeRubric is not None

def test_sandbox_mixin_importable():
    from verifiers.envs.experimental.sandbox_mixin import SandboxMixin
    assert SandboxMixin is not None

def test_create_sandbox_request_importable():
    from verifiers.envs.sandbox_env import CreateSandboxRequest
    assert CreateSandboxRequest is not None
