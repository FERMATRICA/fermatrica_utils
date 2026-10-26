import pytest

from fermatrica_utils.flow import FermatricaUError
from fermatrica_utils.profiling import lprof_to_html


def _make_lprof(tmp_path):
    from line_profiler.line_profiler import LineStats

    src = tmp_path / "mod.py"
    src.write_text("def foo(x):\n    y = x < 3\n    return y\n", encoding="utf-8")

    timings = {
        (str(src), 1, "foo"): [(1, 1, 100), (2, 1, 900)],
    }
    lprof = tmp_path / "out.lprof"
    LineStats(timings, 1e-6).to_file(lprof)
    return lprof


class TestLprofToHtml:

    def test_writes_report(self, tmp_path, if_debug):
        pytest.importorskip("line_profiler")
        lprof = _make_lprof(tmp_path)
        out = lprof_to_html(lprof)
        expected = str(lprof.with_suffix(".html"))

        if if_debug:
            try:
                assert out == expected
            except Exception:
                breakpoint()
        else:
            assert out == expected

        text = lprof.with_suffix(".html").read_text(encoding="utf-8")

        if if_debug:
            try:
                assert "foo" in text
                assert "mod.py" in text
                assert "&lt;" in text
                assert "y = x &lt; 3" in text
                assert "900.0 µs" in text
            except Exception:
                breakpoint()
        else:
            assert "foo" in text
            assert "mod.py" in text
            assert "&lt;" in text
            assert "y = x &lt; 3" in text
            assert "900.0 µs" in text

    def test_custom_out_path(self, tmp_path):
        pytest.importorskip("line_profiler")
        lprof = _make_lprof(tmp_path)
        out = tmp_path / "report.html"
        assert lprof_to_html(lprof, out_path=out) == str(out)
        assert out.exists()

    def test_missing_file(self, tmp_path):
        pytest.importorskip("line_profiler")
        with pytest.raises(FermatricaUError):
            lprof_to_html(tmp_path / "nope.lprof")
