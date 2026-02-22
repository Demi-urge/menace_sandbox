from __future__ import annotations

import json
import sys
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Type, List

from .ipo_bot import IPOBot, ExecutionPlan
try:
    from vector_service.context_builder import ContextBuilder
except ImportError:  # pragma: no cover - fallback when helper missing
    from vector_service.context_builder import ContextBuilder  # type: ignore

from .bot_development_bot import BotDevelopmentBot, BotSpec
from .scalability_assessment_bot import ScalabilityAssessmentBot
from .deployment_bot import DeploymentBot, DeploymentSpec
from .task_handoff_bot import TaskHandoffBot, TaskInfo
from .research_aggregator_bot import ResearchAggregatorBot

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .bot_testing_bot import BotTestingBot


_BOT_TESTING_CLS: Type["BotTestingBot"] | None = None


def _get_bot_testing_bot_class() -> Type["BotTestingBot"]:
    """Return :class:`BotTestingBot` without importing at module import time."""

    global _BOT_TESTING_CLS
    if _BOT_TESTING_CLS is not None:
        return _BOT_TESTING_CLS

    from .bot_testing_bot import BotTestingBot as _BotTestingBot

    _BOT_TESTING_CLS = _BotTestingBot
    return _BOT_TESTING_CLS


def _create_default_tester() -> "BotTestingBot":
    """Instantiate :class:`BotTestingBot` lazily."""

    return _get_bot_testing_bot_class()()

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    name: str
    attempts: int
    deployed: bool
    deploy_id: int | None


class IPOImplementationPipeline:
    """Generate and build bots from an IPO plan through deployment."""

    def __init__(
        self,
        ipo: IPOBot | None = None,
        developer: BotDevelopmentBot | None = None,
        tester: BotTestingBot | None = None,
        scaler: ScalabilityAssessmentBot | None = None,
        deployer: DeploymentBot | None = None,
        handoff: TaskHandoffBot | None = None,
        researcher: ResearchAggregatorBot | None = None,
        *,
        context_builder: ContextBuilder,
        max_attempts: int = 3,
    ) -> None:
        self.ipo = ipo or IPOBot(context_builder=context_builder)
        self.developer = developer or BotDevelopmentBot(
            context_builder=context_builder
        )
        self.tester = tester or _create_default_tester()
        self.scaler = scaler or ScalabilityAssessmentBot()
        self.deployer = deployer or DeploymentBot()
        self.handoff = handoff
        self.researcher = researcher
        self.max_attempts = max_attempts
        self.context_builder = context_builder

    # --------------------------------------------------------------
    def _spec_from_action(self, name: str) -> BotSpec:
        return BotSpec(name=name, purpose=name, functions=["run"])

    def _build_and_test(self, spec: BotSpec) -> bool:
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            path = self.developer.build_bot(
                spec, context_builder=self.context_builder
            )
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
            results = self.tester.run_unit_tests([spec.name])
            errors = [r.error for r in results if not r.passed and r.error]
            if not errors:
                return True
            for err in errors:
                self.developer.errors.append(err)
        return False

    def run(self, blueprint: str) -> List[BuildResult]:
        plan: ExecutionPlan = self.ipo.generate_plan(blueprint)
        if self.handoff:
            tasks = [
                TaskInfo(
                    name=a.bot,
                    dependencies=[],
                    resources={},
                    schedule="once",
                    code="",
                    metadata={},
                )
                for a in plan.actions
            ]
            package = self.handoff.compile(tasks)
            try:
                self.handoff.store_plan(package.tasks)
                self.handoff.send_package(package)
            except Exception:
                logging.getLogger(__name__).exception("handoff failed")

        results: List[BuildResult] = []
        for action in plan.actions:
            spec = self._spec_from_action(action.bot)

            success = self._build_and_test(spec)
            if not success and self.researcher:
                try:
                    self.researcher.process(f"resolve build issues for {spec.name}")
                except Exception:
                    logging.getLogger(__name__).exception("researcher failed")
                success = self._build_and_test(spec)
            if not success:
                results.append(BuildResult(spec.name, self.max_attempts, False, None))
                continue

            attempts = 0
            dep_id: int | None = None
            deployed = False
            while attempts < self.max_attempts and not deployed:
                attempts += 1
                bp = json.dumps({"tasks": [{"name": spec.name}]})
                report = self.scaler.analyse(bp)
                resources = {
                    spec.name: {
                        "cpu": report.tasks[0].cpu,
                        "memory": report.tasks[0].memory,
                    }
                } if report.tasks else {}
                dep_spec = DeploymentSpec(name=spec.name, resources=resources, env={})
                dep_id = self.deployer.deploy(spec.name, [spec.name], dep_spec)
                rec = self.deployer.db.get(dep_id)
                deployed = rec.get("status") == "success"
                if deployed:
                    break
                self.developer.errors.append("deployment failed")

                success = self._build_and_test(spec)
                if not success and self.researcher:
                    try:
                        self.researcher.process(f"resolve deployment issues for {spec.name}")
                    except Exception:
                        logging.getLogger(__name__).exception("researcher deployment help failed")
                    success = self._build_and_test(spec)
                if not success:
                    break

            results.append(BuildResult(spec.name, attempts, deployed, dep_id))
        return results


__all__ = ["BuildResult", "IPOImplementationPipeline"]
