"""Contracts for network-independent small-dataset validation."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NETWORKED_VALIDATION_SOURCES = [
    "scripts/multi_dataset/features/imaging_and_point_source/modeling.py",
    "scripts/weak/start_here.py",
    "scripts/weak/real_data/a2744.py",
    "scripts/weak/features/strong_lensing/a2744.py",
    "scripts/cluster/start_here.py",
    "scripts/cluster/lenstool/data.py",
]


def _test_mentions_not_small_datasets(test: ast.AST) -> bool:
    text = ast.unparse(test)
    return "small_datasets" in text and (
        "not small_datasets" in text or "small_datasets ==" in text and '"0"' in text
    )


def _assert_network_calls_guarded(statements, not_small_guard=False):
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Helper definitions do not execute network I/O by themselves. Their
            # module-level call sites are checked below.
            continue

        if isinstance(statement, ast.If):
            test_text = ast.unparse(statement.test)
            body_not_small = not_small_guard or _test_mentions_not_small_datasets(
                statement.test
            )
            else_not_small = not_small_guard or test_text.strip() == "small_datasets"
            _assert_network_calls_guarded(statement.body, body_not_small)
            _assert_network_calls_guarded(statement.orelse, else_not_small)
            continue

        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                if call.endswith("urlopen") or call == "_download":
                    assert not_small_guard, (
                        f"unguarded runtime network call: {call} at line "
                        f"{getattr(node, 'lineno', '?')}"
                    )


def test_runtime_network_calls_are_excluded_from_small_dataset_mode():
    for relative_path in NETWORKED_VALIDATION_SOURCES:
        source = (ROOT / relative_path).read_text()
        assert "PYAUTO_SMALL_DATASETS" in source, relative_path
        tree = ast.parse(source, filename=relative_path)
        _assert_network_calls_guarded(tree.body)


def test_smoke_and_release_profiles_default_to_small_datasets():
    for relative_path in (
        "config/build/profile_smoke.yaml",
        "config/build/profile_release.yaml",
    ):
        source = (ROOT / relative_path).read_text()
        defaults = source.split("overrides:", 1)[0]
        assert 'PYAUTO_SMALL_DATASETS: "1"' in defaults, relative_path


def test_joint_imaging_small_dataset_has_positive_noise_structure():
    source = (
        ROOT
        / "scripts/multi_dataset/features/imaging_and_point_source/modeling.py"
    ).read_text()

    # The synthetic image deliberately contains a sky gradient. This ensures
    # the existing border-RMS noise-map construction cannot collapse to zero.
    assert "sky = 0.02 + 0.0015 * yy + 0.0010 * xx" in source
    assert "background_rms = float(np.std(clipped))" in source
