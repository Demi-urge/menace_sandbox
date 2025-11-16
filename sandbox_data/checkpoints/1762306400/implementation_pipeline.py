from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING
import subprocess
import json
import logging

from .task_handoff_bot import TaskHandoffBot, TaskInfo, TaskPackage
from .implementation_optimiser_bot import ImplementationOptimiserBot
from .bot_development_bot import BotDevelopmentBot, BotSpec
from contextlib import nullcontext
from .models_repo import clone_to_new_repo, model_build_lock
from dynamic_path_router import resolve_path
from vector_service.context_builder import ContextBuilder

if TYPE_CHECKING:  # pragma: no cover - optional heavy deps
    from .research_aggregator_bot import ResearchAggregatorBot
    from .ipo_bot import IPOBot
else:  # pragma: no cover - avoid heavy import at runtime
    ResearchAggregatorBot = object  # type: ignore
    IPOBot = object  # type: ignore


@dataclass
class PipelineResult:
    """Result object returned by :class:`ImplementationPipeline`."""

    built_files: List[Path]
    package: TaskPackage


class ImplementationPipeline:
    """Coordinate the end-to-end bot implementation workflow.

    The pipeline takes high level task descriptions, forwards them to a
    handoff service for human oversight, fills in any missing metadata using an
    optimiser (and optionally an IPO based planner) and finally uses a
    :class:`BotDevelopmentBot` to produce runnable bot repositories.
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        *,
        handoff: Optional[TaskHandoffBot] = None,
        optimiser: Optional[ImplementationOptimiserBot] = None,
        developer: Optional[BotDevelopmentBot] = None,
        researcher: Optional[ResearchAggregatorBot] = None,
        ipo: Optional[IPOBot] = None,
    ) -> None:
        """Initialize the pipeline.

        Parameters
        ----------
        context_builder : ContextBuilder
            Shared :class:`vector_service.ContextBuilder` instance used to
            assemble local vector database context for prompt generation.
        handoff : TaskHandoffBot, optional
            Service used to store and deliver task packages.
        optimiser : ImplementationOptimiserBot, optional
            Component that adds or corrects metadata for a package.
        developer : BotDevelopmentBot, optional
            Responsible for turning execution plans into code.
        researcher : ResearchAggregatorBot, optional
            Auxiliary agent used when required metadata cannot be derived
            automatically.
        ipo : IPOBot, optional
            Planner used to generate IPO execution plans.
        """
        self.context_builder = context_builder
        self.handoff = handoff or TaskHandoffBot()
        if optimiser is not None:
            try:
                optimiser.context_builder = context_builder  # type: ignore[attr-defined]
            except Exception:
                pass
            self.optimiser = optimiser
        else:
            self.optimiser = ImplementationOptimiserBot(context_builder=context_builder)
        if developer is not None:
            developer.context_builder = context_builder
            self.developer = developer
        else:
            self.developer = BotDevelopmentBot(context_builder=context_builder)
        self.logger = logging.getLogger(self.__class__.__name__)
        if researcher is not None:
            try:
                researcher.context_builder = context_builder  # type: ignore[attr-defined]
            except Exception:
                pass
            self.researcher = researcher
        elif isinstance(ResearchAggregatorBot, type) and ResearchAggregatorBot is not object:
            self.researcher = ResearchAggregatorBot(
                [], context_builder=context_builder
            )  # type: ignore
        else:
            self.researcher = None
        if ipo is not None:
            try:
                ipo.context_builder = context_builder  # type: ignore[attr-defined]
            except Exception:
                pass
            self.ipo = ipo
        elif isinstance(IPOBot, type) and IPOBot is not object:
            self.ipo = IPOBot(context_builder=context_builder)  # type: ignore
        else:
            self.ipo = None

    # --------------------------------------------------------------
    def _missing_info(self, specs: Iterable[BotSpec]) -> bool:
        """Return True if any spec lacks purpose or functions."""
        for s in specs:
            if not s.purpose or not s.functions:
                return True
        return False

    def _package_to_plan(self, pkg: TaskPackage) -> str:
        """Serialise a task package into a JSON plan.

        Parameters
        ----------
        pkg : TaskPackage
            Collection of tasks to encode.

        Returns
        -------
        str
            JSON list describing each task's metadata such as name,
            purpose, language and dependencies.

        Examples
        --------
        >>> plan = pipeline._package_to_plan(TaskPackage(tasks=[]))
        """
        items = []
        for t in pkg.tasks:
            meta = t.metadata or {}
            items.append({
                "name": t.name,
                "purpose": meta.get("purpose", ""),
                "functions": meta.get("functions", []),
                "description": meta.get("description"),
                "function_docs": meta.get("function_docs"),
                "language": meta.get("language", "python"),
                "dependencies": t.dependencies,
                "capabilities": meta.get("capabilities", []),
                "level": meta.get("level", ""),
                "io": meta.get("io") or meta.get("io_format", ""),
            })
        return json.dumps(items)

    def _attempt_handoff(self, package: TaskPackage, attempt: int) -> bool:
        """Attempt to deliver a package to the handoff service.

        Parameters
        ----------
        package : TaskPackage
            The package to send.
        attempt : int
            Attempt counter used for logging.

        Returns
        -------
        bool
            ``True`` if delivery succeeded, ``False`` otherwise.

        Notes
        -----
        All exceptions are caught and logged without being re-raised.
        """
        try:
            self.handoff.send_package(package)
            return True
        except Exception as exc:  # pragma: no cover - network/IO issues
            self.logger.exception("handoff attempt %s failed: %s", attempt, exc)
            return False

    def _apply_ipo_plan(self, package: TaskPackage) -> bool:
        """Apply an IPO generated execution plan to ``package``.

        Parameters
        ----------
        package : TaskPackage
            Task package that will be updated in place.

        Returns
        -------
        bool
            ``True`` when the plan was applied successfully, otherwise
            ``False`` if plan generation failed.

        Examples
        --------
        >>> success = pipeline._apply_ipo_plan(pkg)
        >>> if not success:
        ...     print("IPO plan failed")
        """
        try:
            blueprint = " ".join(t.name for t in package.tasks)
            plan = self.ipo.generate_plan(blueprint)  # type: ignore[attr-defined]
            for act in getattr(plan, "actions", []):
                for t in package.tasks:
                    if t.name == act.bot:
                        meta = t.metadata or {}
                        meta.setdefault("purpose", act.action)
                        meta.setdefault("functions", ["run"])
                        t.metadata = meta
            return True
        except Exception as exc:  # pragma: no cover - IPO may not be available
            self.logger.exception("IPO plan generation failed: %s", exc)
            return False

    def run(self, tasks: Iterable[TaskInfo], model_id: int | None = None) -> PipelineResult:
        """Execute the implementation workflow.

        The method performs the following high level steps:

        1. Compile the incoming ``tasks`` into a :class:`TaskPackage` and
           attempt an initial handoff.
        2. Use :class:`ImplementationOptimiserBot` to infer any missing
           metadata and translate the package into an execution plan that the
           developer understands.
        3. While required information is still missing, retry the handoff and
           (if an :class:`IPOBot` is configured) apply the IPO generated plan.
           Failures in either step are logged; after two consecutive failures an
           exception is raised.
        4. If information is still missing and a researcher is available, the
           researcher is invoked to supply additional context.
        5. Once all bot specifications are complete, the developer builds the
           bot repositories and the resulting file paths are returned.

        Parameters
        ----------
        tasks : Iterable[TaskInfo]
            Iterable of task descriptions for the bots to be built.

        Returns
        -------
        PipelineResult
            Contains the final :class:`TaskPackage` and paths to built files.

        Raises
        ------
        RuntimeError
            If missing information cannot be resolved or bot build fails.
        Exception
            Propagates any unrecoverable errors from handoff or IPO stages.

        Examples
        --------
        >>> pipeline = ImplementationPipeline()
        >>> result = pipeline.run([TaskInfo(name="Demo", dependencies=[],
        ...                                 resources={}, schedule="once",
        ...                                 code="", metadata={})])
        >>> result.built_files
        [PosixPath('Demo/Demo.py')]
        """
        lock = model_build_lock(model_id) if model_id is not None else nullcontext()
        with lock:
            package = self.handoff.compile(list(tasks))
            self.handoff.store_plan(package.tasks)
            try:
                self.handoff.send_package(package)
            except Exception as exc:
                self.logger.exception("initial handoff failed: %s", exc)
                raise

            package = self.optimiser.fill_missing(package)
            self.optimiser.process(package)

            plan_json = self._package_to_plan(package)
            specs = self.developer.parse_plan(plan_json)
            attempt = 0
            send_failures = 0
            plan_failures = 0
            while self._missing_info(specs) and attempt < 2:
                attempt += 1
                if not self._attempt_handoff(package, attempt):
                    send_failures += 1
                    if send_failures >= 2:
                        raise
                if getattr(self, "ipo", None) is not None:
                    if not self._apply_ipo_plan(package):
                        plan_failures += 1
                        if plan_failures >= 2:
                            raise
                package = self.optimiser.fill_missing(package)
                self.optimiser.process(package)
                plan_json = self._package_to_plan(package)
                specs = self.developer.parse_plan(plan_json)

            if self._missing_info(specs) and self.researcher is not None:
                try:
                    self.researcher.process("fill gaps")  # type: ignore[attr-defined]
                except Exception as exc:
                    self.logger.exception("researcher invocation failed: %s", exc)
                    raise
                package = self.optimiser.fill_missing(package)
                plan_json = self._package_to_plan(package)
                specs = self.developer.parse_plan(plan_json)
            if self._missing_info(specs):
                raise RuntimeError("failed to resolve missing bot info")
            try:
                paths = self.developer.build_from_plan(
                    plan_json,
                    model_id=model_id,
                )
            except Exception as exc:
                self.logger.exception("developer build failed: %s", exc)
                raise

            setup_script = str(resolve_path("scripts/setup_tests.sh"))
            for path in paths:
                repo_dir = path.parent
                try:
                    setup_proc = subprocess.run(
                        [setup_script],
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    if setup_proc.stdout:
                        self.logger.info(setup_proc.stdout)
                    if setup_proc.stderr:
                        self.logger.info(setup_proc.stderr)
                    test_proc = subprocess.run(
                        ["pytest", "-q"],
                        cwd=repo_dir,
                        capture_output=True,
                        text=True,
                    )
                    if test_proc.stdout:
                        self.logger.info(test_proc.stdout)
                    if test_proc.stderr:
                        self.logger.info(test_proc.stderr)
                    if test_proc.returncode != 0:
                        raise RuntimeError(
                            f"tests failed for {repo_dir.name}: {test_proc.stdout}{test_proc.stderr}"  # noqa: E501
                        )
                except subprocess.CalledProcessError as exc:
                    self.logger.error(
                        "test setup failed for %s: %s\n%s\n%s",
                        repo_dir.name,
                        exc,
                        exc.stdout,
                        exc.stderr,
                    )
                    raise RuntimeError("test setup failed") from exc

            if model_id is not None:
                try:
                    clone_to_new_repo(model_id)
                except Exception as exc:
                    self.logger.error("model clone failed: %s", exc)

            result = PipelineResult(built_files=paths, package=package)
            return result


__all__ = ["PipelineResult", "ImplementationPipeline"]
