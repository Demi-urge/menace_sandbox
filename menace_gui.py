"""Simple Tkinter GUI for interacting with Menace."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import ttk
from typing import List
from pathlib import Path

try:
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - stub fallback
    ContextBuilder = object  # type: ignore

from .conversation_manager_bot import ConversationManagerBot, ChatGPTClient
from .env_config import OPENAI_API_KEY
from .menace_memory_manager import MenaceMemoryManager
from .report_generation_bot import ReportGenerationBot
from .error_bot import ErrorBot
from .bot_database import BotDB
from .resources_bot import ROIHistoryDB
from .resource_prediction_bot import ResourcePredictionBot, ResourceMetrics
from .resource_allocation_bot import ResourceAllocationBot, AllocationDB
from .scope_utils import Scope, build_scope_clause, apply_scope
try:  # shared GPT memory instance
    from .shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - fallback for flat layout
    from shared_gpt_memory import GPT_MEMORY_MANAGER  # type: ignore


class MenaceGUI(tk.Tk):
    """Main application window with navigation tabs."""

    def __init__(self, *, context_builder: ContextBuilder) -> None:
        super().__init__()
        self.title("Menace Interface")
        self.geometry("600x400")

        self.memory = MenaceMemoryManager()
        self.report_bot = ReportGenerationBot()
        self.chatgpt_enabled = bool(OPENAI_API_KEY)
        self.context_builder = context_builder
        self._log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._max_log_lines = 1000
        self._log_file_path = Path(__file__).with_name("menace_gui_logs.txt")
        try:
            self.context_builder.refresh_db_weights()
        except Exception:  # pragma: no cover - log and disable prompts
            logging.exception("refresh_db_weights failed")
            self.chatgpt_enabled = False
        if self.chatgpt_enabled:
            client = ChatGPTClient(
                api_key=OPENAI_API_KEY,
                gpt_memory=GPT_MEMORY_MANAGER,
                context_builder=self.context_builder,
            )
            self.conv_bot = ConversationManagerBot(client, report_bot=self.report_bot)
        else:
            if not OPENAI_API_KEY:
                logging.warning("OPENAI_API_KEY not set. ChatGPT features disabled.")
            else:
                logging.warning(
                    "Context builder unavailable. Prompt-dependent widgets disabled."
                )
            self.conv_bot = None
        self.error_bot = ErrorBot(context_builder=self.context_builder)
        self._setup_widgets()

    # ------------------------------------------------------------------
    def _setup_widgets(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        self.comm_frame = ttk.Frame(self.notebook)
        self.log_frame = ttk.Frame(self.notebook)
        self.stats_frame = ttk.Frame(self.notebook)
        self.overview_frame = ttk.Frame(self.notebook)
        self.chains_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.comm_frame, text="Communication")
        self.notebook.add(self.log_frame, text="Activity Log")
        self.notebook.add(self.stats_frame, text="Statistics")
        self.notebook.add(self.overview_frame, text="Overview")
        self.notebook.add(self.chains_frame, text="Forecast Chains")

        self._init_comm()
        self._init_log()
        self._init_stats()
        self._init_overview()
        self._init_chains()

    # Communication ----------------------------------------------------
    def _init_comm(self) -> None:
        self.conv_entry = ttk.Entry(self.comm_frame)
        self.conv_entry.pack(fill="x", padx=5, pady=5)
        self.conv_btn = ttk.Button(
            self.comm_frame, text="Send", command=self._ask_menace
        )
        self.conv_btn.pack(padx=5, pady=5)
        self.history_box = tk.Listbox(self.comm_frame, height=5)
        self.history_box.pack(fill="x", padx=5, pady=5)
        self.conv_text = tk.Text(self.comm_frame, state="disabled")
        self.conv_text.pack(expand=True, fill="both", padx=5, pady=5)
        if not self.chatgpt_enabled:
            self.conv_entry.configure(state="disabled")
            self.conv_btn.configure(state="disabled")
            self.conv_text.configure(state="normal")
            self.conv_text.insert(
                tk.END,
                "ChatGPT features disabled. Set OPENAI_API_KEY to enable.\n",
            )
            self.conv_text.configure(state="disabled")

    def _ask_menace(self) -> None:
        if not self.chatgpt_enabled or self.conv_bot is None:
            self.conv_text.configure(state="normal")
            self.conv_text.insert(
                tk.END,
                "ChatGPT features disabled. Set OPENAI_API_KEY to enable.\n",
            )
            self.conv_text.configure(state="disabled")
            return
        query = self.conv_entry.get().strip()
        if not query:
            return
        self.conv_entry.delete(0, tk.END)
        response = self.conv_bot.ask(query)
        self.memory.store("conversation", {"q": query, "a": response})
        self.history_box.insert(tk.END, f"{query} -> {response}")
        if self.history_box.size() > 50:
            self.history_box.delete(0)
        self.conv_text.configure(state="normal")
        self.conv_text.insert(tk.END, f"User: {query}\nMenace: {response}\n")
        notes = self.conv_bot.get_notifications()
        for n in notes:
            self.conv_text.insert(tk.END, f"NOTICE: {n}\n")
        self.conv_text.configure(state="disabled")

    # Activity Log -----------------------------------------------------
    def _init_log(self) -> None:
        self.log_text = tk.Text(self.log_frame, state="disabled")
        self.log_text.pack(expand=True, fill="both", padx=5, pady=5)
        self.log_text.tag_configure("info", foreground="white")
        self.log_text.tag_configure("warning", foreground="yellow")
        self.log_text.tag_configure(
            "error", foreground="red", font=("TkDefaultFont", 10, "bold")
        )
        handler = _TkTextHandler(self._log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        )
        self._logger = logging.getLogger(__name__)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self.after(100, self._poll_log_queue)

    def _poll_log_queue(self) -> None:
        """Poll log records from worker threads and update the widget."""

        drained: list[tuple[str, str]] = []
        try:
            while True:
                drained.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        if drained:
            self.log_text.configure(state="normal")
            for tag, message in drained:
                self.log_text.insert(tk.END, message + "\n", tag)
            self._trim_log()
            self.log_text.configure(state="disabled")
            self.log_text.see(tk.END)
            try:
                with self._log_file_path.open("a", encoding="utf-8") as fh:
                    for _, message in drained:
                        fh.write(message + "\n")
            except OSError:
                pass

        self.after(100, self._poll_log_queue)

    def _trim_log(self) -> None:
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines <= self._max_log_lines:
            return
        excess = lines - self._max_log_lines
        self.log_text.delete("1.0", f"{excess + 1}.0")

    # Statistics -------------------------------------------------------
    def _init_stats(self) -> None:
        self.start_entry = ttk.Entry(self.stats_frame)
        self.start_entry.pack(padx=5, pady=5)
        self.start_entry.insert(0, "start YYYY-MM-DD")
        self.end_entry = ttk.Entry(self.stats_frame)
        self.end_entry.pack(padx=5, pady=5)
        self.end_entry.insert(0, "end YYYY-MM-DD")
        self.stat_btn = ttk.Button(
            self.stats_frame, text="Get Report", command=self._refresh_stats
        )
        self.stat_btn.pack(padx=5, pady=5)
        self.stats_text = tk.Text(self.stats_frame, state="disabled")
        self.stats_text.pack(expand=True, fill="both", padx=5, pady=5)
        if not self.chatgpt_enabled:
            self.start_entry.configure(state="disabled")
            self.end_entry.configure(state="disabled")
            self.stat_btn.configure(state="disabled")
            self.stats_text.configure(state="normal")
            self.stats_text.insert(
                tk.END,
                "ChatGPT features disabled. Set OPENAI_API_KEY to enable.\n",
            )
            self.stats_text.configure(state="disabled")

    def _refresh_stats(self) -> None:
        if not self.chatgpt_enabled or self.conv_bot is None:
            self.stats_text.configure(state="normal")
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert(
                tk.END,
                "ChatGPT features disabled. Set OPENAI_API_KEY to enable.\n",
            )
            self.stats_text.configure(state="disabled")
            return
        start = self.start_entry.get().strip() or None
        end = self.end_entry.get().strip() or None
        report = self.conv_bot.request_report(start=start, end=end)
        text = Path(report).read_text()
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, text)
        self.stats_text.configure(state="disabled")

    # Overview ---------------------------------------------------------
    def _init_overview(self) -> None:
        self.overview = ttk.Treeview(
            self.overview_frame, columns=("roi", "alloc")
        )
        self.overview.heading("#0", text="Bot")
        self.overview.heading("roi", text="ROI")
        self.overview.heading("alloc", text="Resources")
        self.overview.pack(expand=True, fill="both", padx=5, pady=5)
        self.refresh_btn = ttk.Button(
            self.overview_frame, text="Refresh", command=self._refresh_overview
        )
        self.refresh_btn.pack(padx=5, pady=5)
        self._refresh_overview()

    def _refresh_overview(self) -> None:
        self.overview.delete(*self.overview.get_children())
        try:
            bot_db = BotDB()
            roi_db = ROIHistoryDB()
            pred_bot = ResourcePredictionBot()
            alloc_bot = ResourceAllocationBot(
                AllocationDB(), context_builder=self.context_builder
            )

            records = bot_db.fetch_all()
            metrics: dict[str, ResourceMetrics] = {}
            for rec in records:
                name = rec.get("name", "")
                metrics[name] = pred_bot.predict(name)
            scores = alloc_bot.evaluate(metrics)

            for name, m in metrics.items():
                roi_df = roi_db.history(name)
                if not roi_df.empty:
                    latest = float(roi_df.tail(1)["roi"].values[0])
                else:
                    latest = scores.get(name, 0.0)
                res = f"CPU:{m.cpu:.1f} MEM:{m.memory:.1f}"
                self.overview.insert(
                    "", "end", text=name, values=(f"{latest:.2f}", res)
                )
        except Exception:
            bots: List[str] = [
                "chatgpt_research_bot",
                "data_bot",
                "structural_evolution_bot",
            ]
            for b in bots:
                self.overview.insert("", "end", text=b, values=("0.0", "0"))

    # Forecast Chains --------------------------------------------------
    def _init_chains(self) -> None:
        self.chain_bot_var = tk.StringVar()
        self.chain_menu = ttk.OptionMenu(
            self.chains_frame, self.chain_bot_var, "", command=self._show_chain
        )
        self.chain_menu.pack(fill="x", padx=5, pady=5)
        self.chain_text = tk.Text(self.chains_frame, state="disabled")
        self.chain_text.pack(expand=True, fill="both", padx=5, pady=5)
        self.refresh_chains_btn = ttk.Button(
            self.chains_frame, text="Refresh", command=self._refresh_chains
        )
        self.refresh_chains_btn.pack(padx=5, pady=5)
        self._refresh_chains()

    def _refresh_chains(self) -> None:
        bots = list(self.error_bot.last_forecast_chains.keys())
        menu = self.chain_menu["menu"]
        menu.delete(0, "end")
        if not bots:
            self.chain_bot_var.set("")
            return
        for b in bots:
            menu.add_command(label=b, command=lambda x=b: self.chain_bot_var.set(x))
        self.chain_bot_var.set(bots[0])
        self._show_chain(bots[0])

    def _show_chain(self, bot: str) -> None:
        chain = self.error_bot.last_forecast_chains.get(bot, [])
        tel = self._bot_telemetry(bot)
        self.chain_text.configure(state="normal")
        self.chain_text.delete("1.0", tk.END)
        if chain:
            self.chain_text.insert(tk.END, "Chain: " + " -> ".join(chain) + "\n")
        else:
            self.chain_text.insert(tk.END, "No chain available\n")
        if tel:
            self.chain_text.insert(tk.END, "Telemetry:\n")
            for t in tel:
                self.chain_text.insert(
                    tk.END,
                    f"{t['error_type']}: {t['count']} (success {t['success_rate']:.2f})\n",
                )
        self.chain_text.configure(state="disabled")

    def _bot_telemetry(
        self,
        bot: str,
        *,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> list[dict[str, float]]:
        menace_id = self.error_bot.db._menace_id(source_menace_id)
        clause, params = build_scope_clause("telemetry", Scope(scope), menace_id)
        base = (
            "SELECT error_type,"
            "       COUNT(*) as c,"
            "       AVG(CASE WHEN resolution_status='successful' THEN 1.0 ELSE 0.0 END) as rate"
            " FROM telemetry"
            " WHERE bot_id=?"
        )
        query = apply_scope(base, clause)
        query += " GROUP BY error_type ORDER BY c DESC LIMIT 5"
        cur = self.error_bot.db.conn.execute(query, [bot, *params])
        rows = cur.fetchall()
        return [
            {
                "error_type": r[0] if r[0] is not None else "",
                "count": float(r[1]),
                "success_rate": float(r[2] or 0.0),
            }
            for r in rows
        ]


class _TkTextHandler(logging.Handler):
    """Queue-based handler that hands Tk log rendering off to the UI thread."""

    _LEVEL_TAGS: tuple[tuple[int, str], ...] = (
        (logging.ERROR, "error"),
        (logging.WARNING, "warning"),
    )

    def __init__(self, log_queue: queue.Queue[tuple[str, str]]) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return

        tag = "info"
        for level, mapped_tag in self._LEVEL_TAGS:
            if record.levelno >= level:
                tag = mapped_tag
                break
        self._queue.put((tag, message))


__all__ = ["MenaceGUI"]
