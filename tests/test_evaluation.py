from failure2fix.envs import ToyReachEnv
from failure2fix.evaluation import Condition, evaluate_condition, scalar_bool, summarize


def test_toy_oracle_smoke_clean_and_shifted():
    clean = evaluate_condition(
        ToyReachEnv, "toy", "toy-oracle", Condition("clean"), seeds=[1, 2, 3, 4, 5], max_steps=48
    )
    shifted = evaluate_condition(
        ToyReachEnv,
        "toy",
        "toy-oracle",
        Condition("shifted", obs_noise=0.1, action_delay=2),
        seeds=[1, 2, 3, 4, 5],
        max_steps=48,
    )
    assert len(clean) == len(shifted) == 5
    assert all(episode.success is True for episode in clean)
    assert {r.seed for r in clean} == {r.seed for r in shifted}
    assert len(summarize(clean + shifted)) == 2


def test_unknown_success_is_not_inferred_from_termination():
    assert scalar_bool(None) is None
    assert scalar_bool([True, False]) is None
    assert scalar_bool([True]) is True


def test_repeated_seed_is_reproducible():
    a = evaluate_condition(
        ToyReachEnv,
        "toy",
        "toy-oracle",
        Condition("shifted", obs_noise=0.1, action_delay=3),
        seeds=[66],
        max_steps=48,
    )
    b = evaluate_condition(
        ToyReachEnv,
        "toy",
        "toy-oracle",
        Condition("shifted", obs_noise=0.1, action_delay=3),
        seeds=[66],
        max_steps=48,
    )
    assert a == b
