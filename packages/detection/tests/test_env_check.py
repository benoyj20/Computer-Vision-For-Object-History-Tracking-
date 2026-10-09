import pytest

from objhist_detection import env_check


def test_report_marks_device_fields_unavailable_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(env_check.torch.cuda, "is_available", lambda: False)

    report = env_check.collect_report()

    assert report.cuda_available is False
    assert report.device_name is None
    assert report.device_memory_gib is None


def test_require_cuda_fails_without_a_device(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(env_check.torch.cuda, "is_available", lambda: False)

    assert env_check.main(["--require-cuda"]) == 1
    assert "--extra cu130" in capsys.readouterr().err


def test_report_without_requirement_succeeds_on_any_device() -> None:
    assert env_check.main([]) == 0


@pytest.mark.gpu
def test_cuda_device_is_usable() -> None:
    if not env_check.torch.cuda.is_available():
        pytest.skip("no CUDA device")

    report = env_check.collect_report()

    assert report.device_name
    assert env_check.torch.ones(1, device="cuda").item() == 1
