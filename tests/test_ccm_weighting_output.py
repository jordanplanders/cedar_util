from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cedarkit.utils.experiments import ccm


@pytest.mark.parametrize(
    ("weighted", "expected_scheme"),
    [(True, "exp"), (False, "equal")],
)
def test_run_experiment_records_canonical_weighting_scheme(
    tmp_path,
    monkeypatch,
    weighted,
    expected_scheme,
):
    ccm_output = SimpleNamespace(CrossMapList=[object()])
    monkeypatch.setattr(ccm.pe, "CCM", lambda **kwargs: ccm_output)
    monkeypatch.setattr(
        ccm.po,
        "unpack_ccm_output",
        lambda output: pd.DataFrame({"rho": [0.5]}),
    )
    monkeypatch.setattr(
        ccm.po,
        "add_meta_data",
        lambda output, frame, train_ind_i, train_ind_f, lag: frame,
    )
    monkeypatch.setattr(ccm, "log_line", lambda *args, **kwargs: None)

    run_config = SimpleNamespace(
        time_var="time",
        file_path=Path(tmp_path / "output.csv"),
        pset_id="params",
        col_var_id="A",
        target_var_id="B",
        E=2,
        tau=1,
        lag=0,
        knn=3,
        Tp=0,
        tp=0,
        sample=4,
        weighted=weighted,
        surr_var="neither",
        surr_num=0,
        train_ind_i=0,
        train_ind_f=3,
        pred_num=None,
        df=pd.DataFrame({"time": [0, 1], "A": [0.0, 1.0], "B": [1.0, 0.0]}),
        exclusion_radius=1,
        col_var="A",
        target_var="B",
        libsizes="4 4 1",
        embedded=False,
        cpus=1,
        noTime=False,
        self_predict=False,
        output_path=tmp_path,
    )

    output, _ = ccm.run_experiment((run_config, None, 0))

    assert set(output["weighting_scheme"]) == {expected_scheme}
