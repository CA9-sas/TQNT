"""Tests pour le framework de validation predictive."""
import numpy as np
import pytest
from theory.validation import (
    split_calibration_prediction, optimize_eta, predict_masses,
    validation_report, count_degrees_of_freedom, DEFAULT_CALIBRATION,
)
from theory.multi_knot import SM_ASSIGNMENTS, ETA


class TestSplitCalibrationPrediction:
    """Test separation calibration/prediction."""

    def test_default_split(self):
        cal, pred = split_calibration_prediction()
        assert len(cal) == 3
        assert len(pred) == 10
        cal_names = {p["name"] for p in cal}
        assert cal_names == {"e-", "mu-", "W+"}

    def test_custom_split(self):
        cal, pred = split_calibration_prediction(["e-", "t"])
        assert len(cal) == 2
        assert len(pred) == 11

    def test_no_overlap(self):
        cal, pred = split_calibration_prediction()
        cal_names = {p["name"] for p in cal}
        pred_names = {p["name"] for p in pred}
        assert cal_names & pred_names == set()


class TestOptimizeEta:
    """Test eta optimization."""

    def test_eta_close_to_global(self):
        """eta from default calibration should be close to global ETA."""
        cal, _ = split_calibration_prediction()
        eta = optimize_eta(cal)
        rel_diff = abs(eta - ETA) / ETA
        assert rel_diff < 0.05, f"eta={eta:.6f} vs ETA={ETA:.6f}"

    def test_eta_stable_across_splits(self):
        """eta should be stable (±5%) regardless of calibration set."""
        splits = [
            ["e-", "mu-", "W+"],
            ["e-", "tau-", "Z0"],
            ["u", "b", "H"],
            ["d", "c", "t"],
        ]
        etas = []
        for cal_names in splits:
            cal, _ = split_calibration_prediction(cal_names)
            etas.append(optimize_eta(cal))
        etas = np.array(etas)
        mean_eta = np.mean(etas)
        for eta in etas:
            assert abs(eta - mean_eta) / mean_eta < 0.05, (
                f"eta={eta:.6f} deviates >5% from mean={mean_eta:.6f}"
            )


class TestPredictMasses:
    """Test mass predictions."""

    def test_prediction_errors_under_5pct(self):
        """All predicted masses should be within 5% of SM values."""
        cal, pred = split_calibration_prediction()
        eta = optimize_eta(cal)
        results = predict_masses(eta, pred)
        for r in results:
            assert r["error_pct"] < 5.0, (
                f"{r['name']}: error={r['error_pct']:.2f}% exceeds 5%"
            )

    def test_predictions_are_positive(self):
        cal, pred = split_calibration_prediction()
        eta = optimize_eta(cal)
        results = predict_masses(eta, pred)
        for r in results:
            assert r["m_predicted"] > 0


class TestValidationReport:
    """Test full validation report."""

    def test_report_structure(self):
        report = validation_report()
        assert "eta" in report
        assert "predictions" in report
        assert "chi2_log" in report
        assert "max_error_pct" in report
        assert "dof" in report

    def test_report_max_error(self):
        report = validation_report()
        assert report["max_error_pct"] < 5.0

    def test_report_chi2_reasonable(self):
        report = validation_report()
        # chi2_log should be small (sum of log(ratio)^2)
        assert report["chi2_log"] < 0.1


class TestDegreesOfFreedom:
    """Test counting degrees of freedom."""

    def test_dof_count(self):
        dof = count_degrees_of_freedom()
        assert dof["n_particles"] == 13
        assert dof["n_continuous_params"] == 2
        assert dof["n_predictions"] == 11
