from analysis.revision import units_vignette_split as U


def test_explanation_scenarios_from_prompts():
    assert U.explanation_scenarios() == {"pediatric-burn-patient", "middle-aged-woman-cancer-surgery"}


def test_pooled_all_matches_published_text_only_test():
    d = U.compute()
    r = d[(d.score == "nielsen") & (d.group == "all_scenarios") & (d.model == "POOLED")].iloc[0]
    assert abs(r.wilcoxon_p_tight - 0.17) < 0.01
