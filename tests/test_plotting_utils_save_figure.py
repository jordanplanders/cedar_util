from pathlib import Path

from cedarkit.utils.plotting import save_figure


class FakeFigure:
    def __init__(self):
        self.calls = []

    def savefig(self, path, **kwargs):
        self.calls.append((Path(path), kwargs))


def test_save_figure_dry_run_only_prints_local_path(tmp_path, capsys):
    fig = FakeFigure()

    paths = save_figure(fig, tmp_path / "figures", stem="map")

    assert set(paths) == {"local"}
    assert paths["local"].name.startswith("map_")
    assert fig.calls == []
    assert not (tmp_path / "figures").exists()
    assert "Would save local:" in capsys.readouterr().out


def test_save_figure_uses_environment_manuscript_directory(tmp_path, monkeypatch):
    fig = FakeFigure()
    manuscript_dir = tmp_path / "manuscript" / "figures"
    monkeypatch.setenv("MANUSCRIPT_FIGURES_DIR", str(manuscript_dir))

    paths = save_figure(fig, tmp_path / "analysis_figures", stem="map", dry_run=False)

    assert paths["manuscript_tagged"].parent == manuscript_dir / "analysis_figures"
    assert paths["manuscript_canonical"] == manuscript_dir / "map.pdf"
    assert len(fig.calls) == 3
    assert fig.calls[1][1]["dpi"] == 1000
