from pathlib import Path

import pytest
from pydantic import ValidationError

from hybrid_rag.config import CONFIG_ENV_VAR, Settings

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests independent of the developer's real .env and environment."""
    monkeypatch.chdir(tmp_path)  # no .env in the working directory
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "QDRANT_API_KEY", CONFIG_ENV_VAR):
        monkeypatch.delenv(var, raising=False)


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_config_yaml_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(REPO_CONFIG))
    settings = Settings()

    assert settings.qdrant.collection == "documents"
    assert [t.provider for t in settings.llm.answer] == ["openai", "anthropic"]
    assert [t.provider for t in settings.llm.query_rewrite] == ["openai", "anthropic"]


def test_defaults_without_yaml_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "missing.yaml"))
    settings = Settings()

    assert settings.embedding.model == "BAAI/bge-m3"
    assert settings.anthropic_api_key is None


def test_env_var_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _write_yaml(tmp_path / "c.yaml", "qdrant:\n  url: http://from-yaml:6333\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
    monkeypatch.setenv("HYBRID_RAG_QDRANT__URL", "http://from-env:6333")

    assert Settings().qdrant.url == "http://from-env:6333"


def test_api_keys_read_from_standard_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CONFIG_ENV_VAR, str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = Settings()

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test"
    assert "sk-ant-test" not in repr(settings)  # secrets are masked


def test_invalid_chunking_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    yaml_text = "chunking:\n  target_tokens: 100\n  overlap_tokens: 100\n"
    cfg = _write_yaml(tmp_path / "c.yaml", yaml_text)
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))

    with pytest.raises(ValidationError):
        Settings()


def test_unknown_provider_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _write_yaml(tmp_path / "c.yaml", "llm:\n  answer:\n    - {provider: nope, model: x}\n")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))

    with pytest.raises(ValidationError):
        Settings()


def test_tracing_vars_exported_from_dotenv(tmp_path: Path) -> None:
    from hybrid_rag.config import export_tracing_env

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "LANGSMITH_TRACING=true\nLANGSMITH_PROJECT=from-dotenv\nOPENAI_API_KEY=sk-x\n",
        encoding="utf-8",
    )
    environ = {"LANGSMITH_PROJECT": "from-shell"}  # never touch the real os.environ here

    exported = export_tracing_env(dotenv, environ)

    assert exported == ["LANGSMITH_TRACING"]
    assert environ["LANGSMITH_TRACING"] == "true"
    assert environ["LANGSMITH_PROJECT"] == "from-shell"  # the shell wins
    assert "OPENAI_API_KEY" not in environ  # only tracing settings are exported
