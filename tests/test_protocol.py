from robusthw.experiments.protocol import build_plan


def test_main_grid_counts_and_signal_budget_coverage():
    plan = build_plan("main")
    assert len(plan) == 600 and sum(len(item.budgets) for item in plan) == 1800
    assert {item.s for item in plan} == {2, 4, 8}
    assert all(item.budgets == (item.s // 2, item.s, 2 * item.s) for item in plan)
    assert len({item.seed for item in plan}) == 600
    assert len(build_plan("anchor")) == 500


def test_shift_reuses_main_data_without_new_sampling():
    main = {item.independent_data_id: item for item in build_plan("main")}
    shift = build_plan("shift")
    assert len(shift) == 120
    for item in shift:
        source = main[item.independent_data_id]
        assert source.seed == item.seed and item.budgets == (4,)
        assert (item.D, item.n, item.s) == (50, 20, 4)
        assert not item.metadata["refit"]


def test_public_grid_deduplicates_center_and_nests_split_seeds():
    plan = build_plan("public")
    tabular = [item for item in plan if item.kind == "tabular"]
    har = [item for item in plan if item.kind == "har"]
    assert len(tabular) == 90 and len(har) == 10
    assert sum(len(item.budgets) for item in tabular) == 180
    assert sum(len(item.budgets) for item in har) == 30
    for name in ("wdbc", "ionosphere", "sonar"):
        case = [item for item in tabular if item.model == name and item.metadata["repeat"] == 0]
        assert len(case) == 3 and len({item.seed for item in case}) == 1
        assert next(item for item in case if item.n == 20).budgets == (1, 2, 4, 8)


def test_search_and_ablation_counts_keep_timing_repeats_separate():
    assert len(build_plan("small")) == 40
    scaling = build_plan("scaling")
    assert len(scaling) == 200
    centers = [item for item in scaling if item.metadata["timing_repeats"] == 3]
    assert len(centers) == 20
    assert all((item.D, item.n, item.budgets) == (50, 20, (4,)) for item in centers)
    ablation = build_plan("ablation")
    assert len(ablation) == 20
    assert (
        sum(
            len(item.metadata["radius_factors"]) * len(item.metadata["radius_methods"])
            for item in ablation
        )
        == 120
    )
    ids = {item.independent_data_id for item in build_plan("main")}
    assert not ids & {item.independent_data_id for item in ablation}
    assert len(build_plan("development")) == 10


def test_installed_configuration_fallback_uses_single_packaged_source(tmp_path, monkeypatch):
    import shutil

    from robusthw.experiments import protocol

    source = protocol.CONFIG_ROOT / "development.json"
    target = tmp_path / "share" / "robusthw" / "configs"
    target.mkdir(parents=True)
    shutil.copy2(source, target / source.name)
    monkeypatch.setattr(protocol, "CONFIG_ROOT", tmp_path / "missing-repository")
    monkeypatch.setattr(protocol.sys, "prefix", str(tmp_path))
    assert len(protocol.build_plan("development")) == 10
