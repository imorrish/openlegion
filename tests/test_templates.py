"""Tests for extended template system — template application, permissions, workflows."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from src.cli.config import (
    AGENTS_FILE,
    PERMISSIONS_FILE,
    PROJECT_ROOT,
    _add_agent_permissions,
    _add_agent_to_config,
    _apply_template,
)


class _TempConfigMixin:
    """Mixin that redirects config files to a temp directory."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_agents = AGENTS_FILE
        self._orig_perms = PERMISSIONS_FILE
        self._orig_root = PROJECT_ROOT
        # Monkey-patch module-level paths
        import src.cli.config as cfg_mod
        self._agents_path = Path(self._tmpdir) / "config" / "agents.yaml"
        self._perms_path = Path(self._tmpdir) / "config" / "permissions.json"
        self._agents_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_mod.AGENTS_FILE = self._agents_path
        cfg_mod.PERMISSIONS_FILE = self._perms_path
        cfg_mod.PROJECT_ROOT = Path(self._tmpdir)
        # Initialize empty permissions
        self._perms_path.write_text(json.dumps({"permissions": {}}, indent=2))

    def teardown_method(self):
        import src.cli.config as cfg_mod
        cfg_mod.AGENTS_FILE = self._orig_agents
        cfg_mod.PERMISSIONS_FILE = self._orig_perms
        cfg_mod.PROJECT_ROOT = self._orig_root
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class TestAddAgentToConfig(_TempConfigMixin):
    def test_basic_fields(self):
        _add_agent_to_config("alice", "researcher", "openai/gpt-4o")
        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["agents"]["alice"]["role"] == "researcher"
        assert cfg["agents"]["alice"]["model"] == "openai/gpt-4o"

    def test_initial_instructions(self):
        _add_agent_to_config("bob", "engineer", "openai/gpt-4o", initial_instructions="Build things")
        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["agents"]["bob"]["initial_instructions"] == "Build things"

    def test_initial_soul(self):
        _add_agent_to_config("carol", "writer", "openai/gpt-4o", initial_soul="You are creative.")
        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["agents"]["carol"]["initial_soul"] == "You are creative."

    def test_initial_heartbeat(self):
        _add_agent_to_config("dave", "monitor", "openai/gpt-4o", initial_heartbeat="Check alerts.")
        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["agents"]["dave"]["initial_heartbeat"] == "Check alerts."

    def test_thinking(self):
        _add_agent_to_config("eve", "analyst", "openai/gpt-4o", thinking="medium")
        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["agents"]["eve"]["thinking"] == "medium"

    def test_budget(self):
        _add_agent_to_config("frank", "scout", "openai/gpt-4o", budget={"daily_usd": 5.0})
        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["agents"]["frank"]["budget"]["daily_usd"] == 5.0

    def test_empty_fields_not_written(self):
        """Empty optional fields should not appear in agents.yaml."""
        _add_agent_to_config("greg", "helper", "openai/gpt-4o")
        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        agent = cfg["agents"]["greg"]
        assert "initial_soul" not in agent
        assert "initial_heartbeat" not in agent
        assert "thinking" not in agent
        assert "budget" not in agent


class TestAddAgentPermissions(_TempConfigMixin):
    def test_default_permissions(self):
        """Without template permissions, defaults are used."""
        _add_agent_to_config("alice", "researcher", "openai/gpt-4o")
        _add_agent_permissions("alice")
        with open(self._perms_path) as f:
            perms = json.load(f)
        alice = perms["permissions"]["alice"]
        assert alice["allowed_apis"] == ["llm"]
        assert alice["blackboard_read"] == []
        assert alice["blackboard_write"] == []

    def test_template_permissions_merged(self):
        """Template permissions are merged into defaults."""
        _add_agent_to_config("bob", "engineer", "openai/gpt-4o")
        _add_agent_permissions("bob", permissions={
            "blackboard_read": ["tasks/*", "reviews/*"],
            "blackboard_write": ["tasks/*"],
            "can_publish": ["task_complete"],
            "can_subscribe": ["tasks_ready"],
        })
        with open(self._perms_path) as f:
            perms = json.load(f)
        bob = perms["permissions"]["bob"]
        assert "tasks/*" in bob["blackboard_read"]
        assert "reviews/*" in bob["blackboard_read"]
        assert "tasks/*" in bob["blackboard_write"]
        assert "task_complete" in bob["can_publish"]
        assert "tasks_ready" in bob["can_subscribe"]
        # Defaults still present
        assert "llm" in bob["allowed_apis"]

    def test_template_permissions_merge_with_collab_defaults(self):
        """Template permissions add to collaboration defaults, not replace."""
        _add_agent_to_config("carol", "writer", "openai/gpt-4o")
        _add_agent_permissions("carol", permissions={
            "can_publish": ["draft_ready"],
        })
        with open(self._perms_path) as f:
            perms = json.load(f)
        carol = perms["permissions"]["carol"]
        # Both collab default "*" and template "draft_ready" should be present
        assert "draft_ready" in carol["can_publish"]


class TestApplyTemplate(_TempConfigMixin):
    def test_creates_agents_with_new_fields(self):
        tpl = {
            "agents": {
                "scout": {
                    "role": "research_scout",
                    "model": "{default_model}",
                    "instructions": "Find sources.",
                    "soul": "You are curious.",
                    "heartbeat": "Check news.",
                    "thinking": "medium",
                    "budget": {"daily_usd": 5.0, "monthly_usd": 100.0},
                },
            },
        }
        with patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
            "collaboration": True,
        }):
            created = _apply_template("test-tpl", tpl)

        assert "scout" in created
        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        scout = cfg["agents"]["scout"]
        assert scout["initial_instructions"] == "Find sources."
        assert scout["initial_soul"] == "You are curious."
        assert scout["initial_heartbeat"] == "Check news."
        assert scout["thinking"] == "medium"
        assert scout["budget"]["daily_usd"] == 5.0

    def test_sets_permissions_from_template(self):
        tpl = {
            "agents": {
                "analyst": {
                    "role": "analyst",
                    "model": "{default_model}",
                    "permissions": {
                        "blackboard_read": ["data/*"],
                        "blackboard_write": ["analysis/*"],
                        "can_publish": ["analysis_ready"],
                        "can_subscribe": ["data_ready"],
                    },
                },
            },
        }
        with patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
            "collaboration": True,
        }):
            _apply_template("test-tpl", tpl)

        with open(self._perms_path) as f:
            perms = json.load(f)
        analyst = perms["permissions"]["analyst"]
        assert "data/*" in analyst["blackboard_read"]
        assert "analysis/*" in analyst["blackboard_write"]
        assert "analysis_ready" in analyst["can_publish"]
        assert "data_ready" in analyst["can_subscribe"]

    def test_creates_workflow_file(self):
        tpl = {
            "agents": {
                "worker": {
                    "role": "worker",
                    "model": "{default_model}",
                },
            },
            "workflow": {
                "name": "test-pipeline",
                "trigger": "manual",
                "timeout": 300,
                "steps": [
                    {"id": "step1", "agent": "worker", "task_type": "work", "timeout": 120},
                ],
            },
        }
        with patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
            "collaboration": True,
        }):
            _apply_template("test-tpl", tpl)

        wf_path = Path(self._tmpdir) / "config" / "workflows" / "test-pipeline.yaml"
        assert wf_path.exists()
        with open(wf_path) as f:
            wf = yaml.safe_load(f)
        assert wf["name"] == "test-pipeline"
        assert wf["steps"][0]["agent"] == "worker"

    def test_no_workflow_when_not_defined(self):
        tpl = {
            "agents": {
                "helper": {
                    "role": "helper",
                    "model": "{default_model}",
                },
            },
        }
        with patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
            "collaboration": True,
        }):
            _apply_template("test-tpl", tpl)

        wf_dir = Path(self._tmpdir) / "config" / "workflows"
        # Workflow dir may or may not exist, but no yaml files
        if wf_dir.exists():
            assert list(wf_dir.glob("*.yaml")) == []

    def test_model_substitution(self):
        tpl = {
            "agents": {
                "agent1": {
                    "role": "helper",
                    "model": "{default_model}",
                },
            },
        }
        with patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "anthropic/claude-sonnet-4-6"},
            "agents": {},
            "collaboration": True,
        }):
            _apply_template("test-tpl", tpl)

        with open(self._agents_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["agents"]["agent1"]["model"] == "anthropic/claude-sonnet-4-6"

    def test_skills_dir_created(self):
        tpl = {
            "agents": {
                "agent1": {
                    "role": "helper",
                    "model": "{default_model}",
                },
            },
        }
        with patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
            "collaboration": True,
        }):
            _apply_template("test-tpl", tpl)

        skills_dir = Path(self._tmpdir) / "skills" / "agent1"
        assert skills_dir.is_dir()


class TestLoadTemplates:
    def test_all_templates_parse(self):
        """All template YAML files in src/templates/ parse without error."""
        from src.cli.config import _load_templates
        templates = _load_templates()
        assert len(templates) >= 6  # starter, devteam, content, sales, deep-research, monitor
        for name, tpl in templates.items():
            assert "agents" in tpl, f"Template '{name}' missing agents key"
            assert "description" in tpl, f"Template '{name}' missing description"

    def test_templates_have_valid_agent_defs(self):
        """Each agent in each template has at least role and model."""
        from src.cli.config import _load_templates
        templates = _load_templates()
        for tpl_name, tpl in templates.items():
            for agent_name, agent_def in tpl.get("agents", {}).items():
                assert "role" in agent_def, f"{tpl_name}/{agent_name} missing role"
                assert "model" in agent_def, f"{tpl_name}/{agent_name} missing model"
