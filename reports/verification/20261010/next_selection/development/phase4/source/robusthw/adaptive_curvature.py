"""Full binary inner evaluations with a small uncertainty-guided proposal pool.

Only evaluated tests update UB. Only one owned tangent is retained per selected
support. A fixed-kernel mean and uncertainty rank proposals; they never fit a test.
"""
from time import perf_counter

import numpy as np

from .fraction_history import EvaluationHistory, proposal_pool


def solve_adaptive(p, *, oracle_class, gate, time_limit, root_rounds, core_point,
                   warm_support, tolerance, relative_tolerance, max_iterations,
                   cut_mode, proposal_count, certify_every, max_refinements,
                   random_state):
    from .solver import SelectionResult, _ratio_master

    if gate != "risk" or cut_mode != "tangent":
        raise ValueError("the full-inner adaptive path uses risk-gate tangents")
    if proposal_count < 1 or certify_every < 1 or max_refinements < 0:
        raise ValueError("invalid adaptive search controls")
    started = perf_counter()
    oracle = oracle_class(p, gate=gate, core_point=core_point)
    history = EvaluationHistory(p, anchor_target=tolerance / 4)
    rng = np.random.default_rng(random_state)
    if warm_support is None:
        means = [q @ x for q, x in zip(p.weights, p.samples)]
        scales = [np.sqrt(q @ ((x - mean)**2))
                  for q, x, mean in zip(p.weights, p.samples, means)]
        screening = np.abs(means[0] - means[1]) / (scales[0] + scales[1] + 1e-12)
        warm_support = tuple(sorted(np.argsort(-screening, kind="stable")[:p.budget]))
    support = tuple(sorted(warm_support))
    if len(support) != p.budget or len(set(support)) != len(support) or any(
            j < 0 or j >= p.p for j in support):
        raise ValueError("warm_support must contain budget distinct valid coordinates")
    best_test, best_support = oracle._constant(support), support
    lower, upper = 0., float(p.B)
    calls = refinements = master_calls = 0
    oracle_seconds = master_seconds = 0.
    retries, retry_log, trace = 0, [], []
    guard, last_certified_count = None, 0
    anchor_target, master_target = tolerance / 4, tolerance / 4
    pool_count = min(int(proposal_count), 8)

    def remaining():
        return max(0., time_limit - (perf_counter() - started))

    def target():
        return tolerance + relative_tolerance * abs(upper)

    def done():
        return upper - lower <= target()

    def finish(status, **detail):
        quantitative = history.quantitative_bound(tolerance, anchor_target)
        return SelectionResult(best_support, float(lower), float(upper), status,
                    perf_counter() - started, calls, len(history.points), trace, best_test,
                    dict(gate=gate, cut_mode="tangent", search="full_inner_uncertainty_v3",
                         root_rounds=0, requested_root_rounds=root_rounds,
                         fractional_root_calls=0, core_point=core_point, root_bound=0.,
                         oracle_seconds=oracle_seconds, master_seconds=master_seconds,
                         master_retries=retries, master_retry_log=retry_log,
                         master_calls=master_calls, proposal_count=pool_count,
                         requested_proposal_count=proposal_count, certify_every=certify_every,
                         transferred_tests=0, reused_fraction_cuts=0,
                         predicted_upper_updates=0, refinements=refinements,
                         cuts_added=len(history.cuts), cut_count=len(history.cuts), fraction_count=len(history.fractions), evaluations=history.records,
                         accurate_support_evaluations=history.accurate_count,
                         anchor_target=anchor_target, master_target=master_target,
                         screening_margin=tolerance / 2, anchor_gaps=history.anchor_gaps,
                         metric_weights=history.metric_weights.tolist(),
                         forecast_model=history.predictor.model_tag,
                         prediction_training_points=len(history.predictor.responses),
                         prediction_factor_updates=history.predictor.factor_updates,
                         prediction_factor_reuses=history.predictor.factor_reuses,
                         prediction_batches=history.predictor.prediction_batches,
                         predicted_points=history.predictor.predicted_points,
                         prediction_seconds=history.predictor.prediction_seconds,
                         forecast_update_seconds=history.predictor.update_seconds,
                         prediction_noise_variance=history.predictor.noise_variance,
                         kernel_length_scale=history.predictor.length_scale,
                         model_reuse=getattr(oracle, "reuse_statistics", {}),
                         **quantitative, **detail))

    def evaluate(z, phase):
        nonlocal calls, refinements, oracle_seconds, upper, best_test, best_support
        z = np.asarray(z, dtype=float)
        key = p.support(z)
        if len(key) != p.budget:
            raise ValueError("the adaptive oracle only receives full-budget binary points")
        if key in history.tests:
            raise ArithmeticError("attempted to reevaluate a completed support")
        before_lower = history.lower(z)
        screen = upper - tolerance / 2
        if before_lower > screen:
            raise ArithmeticError("selected support fails the improvement screen")
        best_local_test = oracle._constant(key)
        best_cut = best_fraction = None
        best_cut_value = -np.inf
        attempts_done, last_status = 0, "time_limit"
        for attempt in range(max_refinements + 1):
            if remaining() <= 0:
                break
            calls += 1
            refinements += int(attempt > 0)
            attempts_done += 1
            tick = perf_counter()
            result = oracle.solve(z, time_limit=remaining())
            oracle_seconds += perf_counter() - tick
            gradient = np.asarray(result.gradient, dtype=float)
            if gradient.shape != (p.p,) or not np.isfinite(gradient).all():
                raise ArithmeticError("invalid oracle gradient")
            if not np.isfinite(result.value) or not np.isfinite(result.upper) or not np.isfinite(
                    result.intercept):
                raise ArithmeticError("nonfinite oracle bounds")
            owned_lower = float(result.intercept + gradient @ z)
            if abs(owned_lower-result.value) > 1e-7 * max(1., p.B):
                raise ArithmeticError("oracle value and owned tangent disagree")
            if result.test is not None:
                if tuple(result.test.support) != key:
                    raise ArithmeticError("oracle test belongs to another support")
                if not np.isfinite(result.test.upper):
                    raise ArithmeticError("oracle returned a nonfinite actual test bound")
                if result.test.upper <= best_local_test.upper:
                    best_local_test = result.test
            if owned_lower > best_local_test.upper + 1e-7 * max(1., p.B):
                raise ArithmeticError("oracle cut exceeds a genuine evaluated test")
            if best_cut is None or owned_lower > best_cut_value:
                best_cut, best_cut_value = (float(result.intercept), gradient.copy()), owned_lower
                cert=result.certificate
                tau=float(cert.get("tau",0.))
                numerator=max(0.,float(cert.get("entropy",0.))-float(cert.get("margin",0.)))
                best_fraction=(numerator,tau,np.asarray(cert["mu"],float).copy()) if tau>0 and numerator>0 else None
            last_status = result.status
            inner_gap = max(0., best_local_test.upper-best_cut_value)
            trace.append(dict(phase="inner_attempt", evaluation_phase=phase,
                        seconds=perf_counter()-started, support=list(key), point=z.tolist(),
                        oracle_lower=float(result.value), oracle_upper=float(result.upper),
                        retained_lower=best_cut_value,
                        retained_actual_upper=float(best_local_test.upper),
                        lower=float(lower), upper=float(upper), anchor_gap=inner_gap,
                        refinement=attempt, status=last_status, cut_added=False))
            if inner_gap <= anchor_target:
                break
            if hasattr(oracle, "tolerance"):
                oracle.tolerance *= .1
        if not attempts_done:
            raise TimeoutError("wall-time budget exhausted before the inner evaluation")
        gap = max(0., float(best_local_test.upper-best_cut_value))
        accurate = gap <= anchor_target
        history.add(z, test=best_local_test, intercept=best_cut[0],
                    gradient=best_cut[1], accurate=accurate, fraction=best_fraction)
        if best_local_test.upper <= upper:
            upper, best_test, best_support = float(best_local_test.upper), best_local_test, key
        trace.append(dict(phase=phase, seconds=perf_counter()-started,
                    support=list(key), point=z.tolist(), oracle_lower=best_cut_value,
                    oracle_upper=float(best_local_test.upper), lower=float(lower),
                    upper=float(upper), anchor_gap=gap, attempts=attempts_done,
                    status="accurate" if accurate else last_status,
                    selection_model_lower=before_lower, screening_threshold=screen,
                    cut_added=True))
        return accurate, last_status

    def master():
        nonlocal lower, master_seconds, master_calls, retries, last_certified_count
        if remaining() <= 0:
            raise TimeoutError("wall-time budget exhausted")
        tick = perf_counter()
        result = _ratio_master(p, history.cuts, history.fractions, remaining(), upper)
        master_seconds += perf_counter() - tick
        master_calls += 1
        count = int(getattr(result, "master_retries", 0))
        retries += count
        retry_log.extend(getattr(result, "master_retry_log", []))
        bound = getattr(result, "mip_dual_bound", None) if result.status in (0, 1) else None
        if result.status == 0 and bound is None:
            bound = result.fun
        if result.status in (0, 1) and bound is not None and np.isfinite(bound):
            margin = 1e-9 * max(1., abs(bound), p.B)
            lower = max(lower, float(bound)-margin)
        trace.append(dict(phase="master", seconds=perf_counter()-started,
                          lower=float(lower), upper=float(upper), status=int(result.status),
                          master_retries=count))
        if lower > upper + 1e-7 * max(1., p.B):
            raise ArithmeticError("global master bound exceeds an evaluated test")
        last_certified_count = len(history.points)
        return result

    def candidates(points):
        seen, eligible = set(), []
        threshold = upper - tolerance / 2
        for z in points:
            z = np.asarray(z, dtype=float)
            key = p.support(z)
            if len(key) != p.budget:
                raise ValueError("proposal must be a full-budget binary selection")
            if key in seen or key in history.tests:
                continue
            seen.add(key)
            model = history.lower(z)
            if model <= threshold:
                eligible.append((z, model))
        if not eligible:
            return [], len(seen)
        means, deviations = history.predict(np.vstack([point for point, _ in eligible]))
        if not np.isfinite(means).all() or not np.isfinite(deviations).all():
            raise ArithmeticError("nonfinite candidate forecast")
        return [(point, model, float(mean), float(sigma))
                for (point, model), mean, sigma in zip(eligible, means, deviations)], len(seen)

    def failed_evaluation(status, key):
        if remaining() <= 0 or status == "time_limit":
            return finish("time_limit", failed_support=list(key))
        if status == "solver_error":
            return finish("oracle_failure", failed_support=list(key))
        return finish("oracle_accuracy_limit", failed_support=list(key))

    try:
        if done():
            return finish("optimal")
        initial = p.mask(support).astype(float)
        accurate, status = evaluate(initial, "initial")
        if not accurate:
            return failed_evaluation(status, support)
        if done():
            return finish("optimal")
        if p.budget in (0, p.p):
            lower = max(lower, history.lower(initial)-1e-9 * max(1., p.B))
            return finish("optimal" if done() else "oracle_accuracy_limit")
        for _ in range(max_iterations):
            if remaining() <= 0:
                return finish("time_limit")
            points = proposal_pool(p, history, best_support, count=pool_count,
                                   rng=rng, guard=guard, remaining=remaining)
            scheduled = len(history.points)-last_certified_count >= certify_every
            choices, candidate_count = ([], 0) if scheduled else candidates(points)
            if scheduled or not choices:
                result = master()
                if done():
                    return finish("optimal")
                if result.status not in (0, 1) or result.x is None:
                    return finish("time_limit" if result.status == 1 else "master_failure")
                raw = result.x[:p.p]
                z_guard = np.rint(raw)
                if np.max(abs(raw-z_guard)) > 1e-5 or z_guard.sum() != p.budget:
                    return finish("master_failure", message="no feasible binary master candidate")
                candidate_gap = max(0., history.lower(z_guard)-lower)
                trace[-1]["candidate_gap"] = candidate_gap
                if candidate_gap > master_target:
                    return finish("time_limit" if result.status == 1 else "master_accuracy_limit",
                                  message="the global candidate lacks the required master accuracy",
                                  candidate_gap=candidate_gap)
                guard = p.support(z_guard)
                points = [z for z in points if p.support(z) != guard][:pool_count-1]
                choices, candidate_count = candidates([*points, z_guard])
            if not choices:
                return finish("master_accuracy_limit",
                              message="no globally certified candidate passes the improvement screen")
            chosen, model, mean, sigma = min(choices, key=lambda item:
                                            (item[2]-item[3], p.support(item[0])))
            min_cut = min(choices, key=lambda item: (item[1], p.support(item[0])))[0]
            distances = [history.distance(chosen, old) for old in history.points]
            trace.append(dict(phase="acquisition", seconds=perf_counter()-started,
                        support=list(p.support(chosen)), lower=float(lower), upper=float(upper),
                        model_lower=model, mean=mean, std=sigma,
                        screening_threshold=upper-tolerance/2, improvement_margin=tolerance/2,
                        score=mean-sigma, candidates=candidate_count,
                        candidate_count=candidate_count, eligible=len(choices),
                        mean_range=[min(item[2] for item in choices), max(item[2] for item in choices)],
                        std_range=[min(item[3] for item in choices), max(item[3] for item in choices)],
                        chosen_differs_from_min_cut=p.support(chosen) != p.support(min_cut),
                        nearest_history_distance=min(distances, default=None)))
            accurate, status = evaluate(chosen, "adaptive")
            if not accurate:
                return failed_evaluation(status, p.support(chosen))
            if done():
                return finish("optimal")
        return finish("iteration_limit")
    except TimeoutError as error:
        return finish("time_limit", message=str(error))
    except (ArithmeticError, RuntimeError) as error:
        return finish("oracle_failure", message=str(error))
