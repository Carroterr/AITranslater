import os
import sqlite3
import tkinter as tk
from bisect import bisect_left
from dataclasses import replace
from pathlib import Path
from queue import Empty
from time import time
from tkinter import font, ttk

from audio import Transcript, list_microphones, list_playback_devices
from deepseek_asr import AudioFormat
from history import ConversationHistory, HistoryEntry
from interpreter import SCENE_SKILLS, Interpreter, InterviewUpdate, Status, Subtitle, TranslationMode, api_key_error


SOURCE_LANGUAGES = {
    "英语": "en",
    "普通话": "zh",
    "日语": "ja",
    "俄语": "ru",
    "阿拉伯语": "ar",
    "印尼语": "id",
    "泰语": "th",
    "越南语": "vi",
}
TARGET_LANGUAGES = {"简体中文": "zh-Hans", "繁體中文": "zh-Hant"}
DEVICES = {"优先 GPU": True, "仅 CPU": False}
DEFAULT_PLAYBACK = "Windows 默认播放设备"
DEFAULT_MICROPHONE = "Windows 默认麦克风"
IDLE_CAPTION = "字幕会显示在这里 · 可拖动"
BACKGROUND = "#15171c"
PANEL_BACKGROUND = "#1e222a"
TEXT = "#f3f5f9"
SECONDARY = "#a7adbb"
FONT = "Microsoft YaHei UI"
WAITING_FOR_QUESTIONS = "等待识别面试问题…"


def caption_text(subtitle):
    if subtitle is None:
        return ""
    text = subtitle.text
    if subtitle.final or text.endswith(("。", "！", "？", ".", "!", "?", "…")):
        return text
    return f"{text}…"


def caption_order(caption):
    return caption.recorded_at or 0, caption.segment


class Overlay:
    def __init__(self, root):
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.attributes("-alpha", 0.94)
        self.window.configure(bg=BACKGROUND)
        self.window.title("AITranslate 悬浮字幕")
        self.window.columnconfigure(0, weight=1)

        self.width = min(1050, int(root.winfo_screenwidth() * 0.82))
        self.panel_width = min(320, round(self.width * 0.34))
        self.current_font = font.Font(root, family=FONT, size=28, weight="bold")
        self.previous_font = font.Font(root, family=FONT, size=17)
        self.previous_source = self._line(SECONDARY, self.previous_font, 0, (8, 0))
        self.previous = self._line(SECONDARY, self.previous_font, 1, (3, 0))
        self.current_source = self._line(SECONDARY, self.previous_font, 2, (8, 5))
        self.current = self._line(TEXT, self.current_font, 3, (3, 10))
        self.question_count = 0
        self.question_panel = tk.Frame(self.window, bg=PANEL_BACKGROUND, width=self.panel_width, height=180)
        self.question_panel.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=(0, 2), pady=2)
        self.question_panel.grid_propagate(False)
        self.question_panel.columnconfigure(0, weight=1)
        self.question_panel.rowconfigure(1, weight=1)
        tk.Label(
            self.question_panel, text="面试问题总结", bg=PANEL_BACKGROUND, fg=TEXT, font=(FONT, 12, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 6))
        self.questions = tk.Text(
            self.question_panel, bg=PANEL_BACKGROUND, fg=TEXT, font=(FONT, 11), wrap="word", relief="flat",
            padx=0, pady=0, borderwidth=0, highlightthickness=0, state="disabled",
        )
        self.questions_scrollbar = ttk.Scrollbar(self.question_panel, orient="vertical", command=self.questions.yview)
        self.questions.configure(yscrollcommand=self.questions_scrollbar.set)
        self.questions.grid(row=1, column=0, sticky="nsew", padx=(14, 8), pady=(0, 8))
        self.questions_scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 6), pady=(0, 8))
        self.question_status = tk.Label(
            self.question_panel, text="开始翻译后提取面试问题。", bg=PANEL_BACKGROUND, fg=SECONDARY,
            font=(FONT, 10), justify="left", wraplength=self.panel_width - 28,
        )
        self.question_status.grid(row=2, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 10))
        self.question_panel.grid_remove()
        self._default_position = True
        self._position_job = None
        self._drag_offset = None
        self.window.bind("<ButtonPress-1>", self._begin_drag)
        self.window.bind("<B1-Motion>", self._drag)
        self.window.bind("<ButtonRelease-1>", self._end_drag)
        self.render([])

    def _line(self, foreground, typeface, row, padding):
        label = tk.Label(
            self.window, bg=BACKGROUND, fg=foreground, font=typeface,
            wraplength=self.width - 48, justify="center", cursor="fleur",
        )
        label.grid(row=row, column=0, sticky="ew", padx=24, pady=padding)
        return label

    def _show_line(self, label, text):
        if text:
            label.configure(text=text)
            label.grid()
        else:
            label.grid_remove()

    def _begin_drag(self, event):
        if event.widget in (self.questions, self.questions_scrollbar):
            return
        self._default_position = False
        self._drag_offset = (
            event.x_root - self.window.winfo_x(),
            event.y_root - self.window.winfo_y(),
        )

    def _drag(self, event):
        if self._drag_offset is not None:
            dx, dy = self._drag_offset
            self.window.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _end_drag(self, _event):
        self._drag_offset = None

    def position_default(self):
        self._default_position = True
        self._schedule_position()

    def _schedule_position(self):
        if self._default_position and self._position_job is None:
            self._position_job = self.window.after_idle(self._place)

    def _place(self):
        self._position_job = None
        if self._default_position:
            width = self.window.winfo_reqwidth()
            height = self.window.winfo_reqheight()
            screen_width = self.window.winfo_screenwidth()
            screen_height = self.window.winfo_screenheight()
            x = (screen_width - width) // 2
            y = max(0, int(screen_height * 0.89) - height)
            self.window.geometry(f"+{x}+{y}")

    def set_size(self, size):
        self.current_font.configure(size=size)
        self.previous_font.configure(size=max(12, round(size * 0.64)))
        self._schedule_position()

    def set_visible(self, visible):
        if visible:
            self.window.deiconify()
        else:
            self.window.withdraw()

    def set_topmost(self, topmost):
        self.window.attributes("-topmost", topmost)

    def set_interview(self, enabled):
        if enabled:
            self.question_panel.grid()
        else:
            self.question_panel.grid_remove()
        wraplength = self.width - 48 - (self.panel_width if enabled else 0)
        for label in (self.previous_source, self.previous, self.current_source, self.current):
            label.configure(wraplength=wraplength)
        self._schedule_position()

    def _show_question_status(self, text):
        if text:
            self.question_status.configure(text=text)
            self.question_status.grid()
        else:
            self.question_status.grid_remove()

    def reset_questions(self, text):
        self.question_count = 0
        self.questions.configure(state="normal")
        self.questions.delete("1.0", "end")
        self.questions.configure(state="disabled")
        self._show_question_status(text)

    def show_questions(self, event):
        if event.questions:
            follow = self.questions.yview()[1] >= 0.99
            self.questions.configure(state="normal")
            for question in event.questions:
                self.question_count += 1
                self.questions.insert("end", f"{self.question_count}. {question}\n\n")
            self.questions.configure(state="disabled")
            if follow:
                self.questions.see("end")
        self._show_question_status(event.warning or (WAITING_FOR_QUESTIONS if not self.question_count else ""))

    def render(self, captions, placeholder=IDLE_CAPTION):
        previous_source, previous = captions[-2] if len(captions) > 1 else ("", None)
        current_source, current = captions[-1] if captions else ("", None)
        self._show_line(self.previous_source, previous_source)
        self._show_line(self.previous, caption_text(previous))
        self._show_line(self.current_source, current_source)
        self.current.configure(fg=TEXT if current and current.final else "#dbe8ff" if current else SECONDARY)
        self._show_line(self.current, caption_text(current) if captions else placeholder)
        self._schedule_position()


class ConversationLog(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.keys = []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.text = tk.Text(
            self, bg=PANEL_BACKGROUND, fg=TEXT, font=(FONT, 11), wrap="word", relief="flat",
            padx=16, pady=8, borderwidth=0, highlightthickness=0, state="disabled",
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text.tag_configure("time", foreground=SECONDARY, font=(FONT, 9), spacing1=12, spacing3=3)
        self.text.tag_configure("source", foreground=SECONDARY, spacing3=2)

    @staticmethod
    def _tag(key):
        session, _, segment = key
        return f"entry_{session}_{segment}"

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if self.keys:
            self.text.tag_delete(*(self._tag(key) for key in self.keys))
            self.keys.clear()
        self.text.configure(state="disabled")

    def update_entries(self, entries):
        follow = not self.keys or self.text.yview()[1] >= 0.999
        self.text.configure(state="normal")
        for entry in sorted(entries, key=lambda item: item.order_key):
            key = entry.order_key
            tag = self._tag(key)
            position = bisect_left(self.keys, key)
            ranges = self.text.tag_ranges(tag)
            if ranges:
                index = str(ranges[0])
                self.text.delete(*ranges)
            else:
                index = "end-1c"
                if position < len(self.keys):
                    index = self.text.tag_ranges(self._tag(self.keys[position]))[0]
                self.keys.insert(position, key)

            parts = [entry.heading + "\n", (tag, "time")]
            if entry.source:
                parts.extend([entry.source + "\n", (tag, "source")])
            if entry.translation:
                parts.extend([entry.translation + "\n", (tag,)])
            self.text.insert(index, *parts)
        self.text.configure(state="disabled")
        if follow:
            self.text.see("end")


class LiveHistoryWindow:
    def __init__(self, root, close):
        self.window = tk.Toplevel(root, bg=BACKGROUND)
        self.window.title("AITranslate · 本轮对话")
        width, height = 600, 440
        x = max(0, root.winfo_screenwidth() - width - 48)
        self.window.geometry(f"{width}x{height}+{x}+40")
        self.window.minsize(360, 235)
        self.window.protocol("WM_DELETE_WINDOW", close)

        content = ttk.Frame(self.window, padding=12)
        content.pack(fill="both", expand=True)
        self.message = ttk.Label(content, style="Hint.TLabel")
        self.message.pack(anchor="w", pady=(0, 7))
        self.log = ConversationLog(content)
        self.log.pack(fill="both", expand=True)

    def refresh_message(self, running):
        if self.log.keys:
            message = "翻译中" if running else "本轮翻译已结束"
        else:
            message = "等待识别语音…" if running else "开始翻译后显示本轮对话。"
        self.message.configure(text=message)


class HistoryWindow:
    PAGE_SIZE = 200

    def __init__(self, root, history, enabled, close):
        self.history = history
        self.enabled = enabled
        self.older_before = None
        self.window = tk.Toplevel(root, bg=BACKGROUND)
        self.window.title("AITranslate · 对话历史")
        self.window.geometry("670x500")
        self.window.minsize(440, 300)
        self.window.protocol("WM_DELETE_WINDOW", close)

        content = ttk.Frame(self.window, padding=16)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)
        self.message = ttk.Label(content, style="Hint.TLabel")
        self.message.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.older = ttk.Button(content, text="加载更早记录", command=self.load_older)
        self.older.grid(row=0, column=1, sticky="e", pady=(0, 8))

        self.log = ConversationLog(content)
        self.log.grid(row=1, column=0, columnspan=2, sticky="nsew")
        ttk.Button(content, text="打开文档文件夹", command=self.open_documents).grid(
            row=2, column=0, columnspan=2, sticky="e", pady=(8, 0),
        )
        self.load_older()

    def refresh_message(self):
        if self.log.keys:
            message = "记录已开启" if self.enabled.get() else "记录已关闭"
        else:
            message = "等待识别语音…" if self.enabled.get() else "在设置中开启「记录对话历史」即可开始记录。"
        self.message.configure(text=message)

    def load_older(self):
        before = self.older_before
        try:
            entries = self.history.recent(self.PAGE_SIZE + 1, before=before)
        except (OSError, sqlite3.Error):
            self.message.configure(text="读取对话历史失败，请重试。")
            return
        has_older = len(entries) > self.PAGE_SIZE
        self.older_before = entries[self.PAGE_SIZE - 1].order_key if has_older else None
        self.older.configure(state="normal" if has_older else "disabled")
        self.update_entries(entries[:self.PAGE_SIZE])
        self.log.text.yview_moveto(0 if before is not None else 1)

    def open_documents(self):
        try:
            directory = self.history.documents_dir
            directory.mkdir(parents=True, exist_ok=True)
            os.startfile(directory)
        except OSError:
            self.message.configure(text="无法打开对话文档文件夹。")

    def update_entries(self, entries):
        self.log.update_entries(entries)
        self.refresh_message()


class App:
    def __init__(self, root):
        self.root = root
        root.title("AITranslate · 实时字幕")
        root.geometry("510x690")
        root.minsize(460, 690)
        root.configure(bg=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.session = None
        self.stopping = False
        self.playback_devices = []
        self.microphone_devices = []
        self.captions = {}
        self.transcripts = {}
        self.live_entries = {}
        self.live_history_window = None
        history_path = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "AITranslate" / "history.sqlite3"
        self.history = ConversationHistory(history_path)
        self.history_window = None
        self.record_history = tk.BooleanVar(value=False)
        self.document_sync_job = None
        self.document_error = ""
        history_error = ""
        try:
            self.record_history.set(self.history.enabled())
        except (OSError, sqlite3.Error):
            history_error = "无法读取对话历史设置。"
        self.api_key = tk.StringVar(value=os.environ.get("DEEPSEEK_API_KEY", ""))
        self.mode = tk.StringVar(value=TranslationMode.LOCAL.value)
        self.scene = tk.StringVar(value="看剧")
        self.source = tk.StringVar(value=next(iter(SOURCE_LANGUAGES)))
        self.target = tk.StringVar(value=next(iter(TARGET_LANGUAGES)))
        self.device = tk.StringVar(value=next(iter(DEVICES)))
        self.audio_format = tk.StringVar(value=AudioFormat.OGG.value)
        self.capture_mic = tk.BooleanVar(value=False)
        self.review_captions = tk.BooleanVar(value=True)
        self.visible = tk.BooleanVar(value=True)
        self.show_source = tk.BooleanVar(value=True)
        self.show_live_history = tk.BooleanVar(value=True)
        self.topmost = tk.BooleanVar(value=True)
        self.size_label = tk.StringVar(value="28 pt")
        self.status = tk.StringVar(value="等待开始")
        self.overlay = Overlay(root)

        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT, font=(FONT, 10))
        style.configure("Title.TLabel", font=(FONT, 22, "bold"))
        style.configure("Hint.TLabel", foreground=SECONDARY, font=(FONT, 9))
        style.configure("TEntry", fieldbackground="#21252e", foreground=TEXT, insertcolor=TEXT)
        style.configure("TCombobox", fieldbackground="#21252e", foreground=TEXT, background="#303746")
        style.map("TCombobox", fieldbackground=[("readonly", "#21252e")])
        style.configure("TButton", background="#272c37", foreground=TEXT, padding=(10, 6), font=(FONT, 10))
        style.map("TButton", background=[("active", "#353d4c")], foreground=[("disabled", SECONDARY)])
        style.configure("Primary.TButton", background="#b7d4ff", foreground="#132237", padding=(12, 8))
        style.map("Primary.TButton", background=[("active", "#d2e4ff"), ("disabled", "#272c37")])
        style.configure("TCheckbutton", background=BACKGROUND, foreground=TEXT, font=(FONT, 10))
        style.map("TCheckbutton", background=[("active", BACKGROUND)])

        content = ttk.Frame(root, padding=(26, 22))
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="实时字幕", style="Title.TLabel").pack(anchor="w")
        ttk.Label(content, text="Windows 系统声音", style="Hint.TLabel").pack(anchor="w", pady=(2, 17))

        ttk.Label(content, text="DeepSeek API Key").pack(anchor="w")
        self.key_entry = ttk.Entry(content, textvariable=self.api_key, show="●", font=(FONT, 11))
        self.key_entry.pack(fill="x", pady=(4, 3), ipady=4)
        ttk.Label(content, text="两种模式都会将所选设备的语音片段发送给 DeepSeek。", style="Hint.TLabel").pack(anchor="w")

        settings = ttk.Frame(content)
        settings.pack(fill="x", pady=(15, 10))
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="翻译模式").grid(row=0, column=0, sticky="w", padx=(0, 16))
        self.mode_selector = ttk.Combobox(
            settings, textvariable=self.mode, values=[mode.value for mode in TranslationMode], state="readonly", width=32,
        )
        self.mode_selector.grid(row=0, column=1, columnspan=2, sticky="w")
        self.mode_selector.bind("<<ComboboxSelected>>", self.set_mode)
        ttk.Label(settings, text="适用场景").grid(row=1, column=0, sticky="w", padx=(0, 16), pady=(8, 0))
        self.scene_selector = ttk.Combobox(
            settings, textvariable=self.scene, values=list(SCENE_SKILLS), state="readonly", width=13,
        )
        self.scene_selector.grid(row=1, column=1, sticky="w", pady=(8, 0))
        self.scene_selector.bind("<<ComboboxSelected>>", self.set_scene)
        ttk.Label(settings, text="翻译原文语言").grid(row=2, column=0, sticky="w", padx=(0, 16), pady=(8, 0))
        self.source_language = ttk.Combobox(
            settings, textvariable=self.source, values=list(SOURCE_LANGUAGES), state="readonly", width=13,
        )
        self.source_language.grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="译文语言").grid(row=3, column=0, sticky="w", padx=(0, 16), pady=(8, 0))
        self.language = ttk.Combobox(
            settings, textvariable=self.target, values=list(TARGET_LANGUAGES), state="readonly", width=13,
        )
        self.language.grid(row=3, column=1, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="本地翻译设备").grid(row=4, column=0, sticky="w", padx=(0, 16), pady=(8, 0))
        self.translation_device = ttk.Combobox(
            settings, textvariable=self.device, values=list(DEVICES), state="readonly", width=13,
        )
        self.translation_device.grid(row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="播放设备").grid(row=5, column=0, sticky="w", padx=(0, 16), pady=(8, 0))
        self.playback_selector = ttk.Combobox(
            settings, values=(DEFAULT_PLAYBACK,), state="readonly", width=16,
            postcommand=self.refresh_playback_devices,
        )
        self.playback_selector.grid(row=5, column=1, columnspan=2, sticky="ew", pady=(8, 0))
        self.playback_selector.current(0)
        self.playback_selector.bind("<<ComboboxSelected>>", lambda _event: self.status.set("等待开始"))
        self.microphone_settings = ttk.Frame(settings)
        self.microphone_settings.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.microphone_settings.columnconfigure(1, weight=1)
        self.microphone_toggle = ttk.Checkbutton(
            self.microphone_settings, text="采集麦克风（B）", variable=self.capture_mic,
            command=self.set_microphone_state,
        )
        self.microphone_toggle.grid(row=0, column=0, sticky="w", padx=(0, 16))
        self.microphone_selector = ttk.Combobox(
            self.microphone_settings, values=(DEFAULT_MICROPHONE,), state="readonly", width=16,
            postcommand=self.refresh_microphones,
        )
        self.microphone_selector.grid(row=0, column=1, sticky="ew")
        self.microphone_selector.current(0)
        self.microphone_selector.bind("<<ComboboxSelected>>", lambda _event: self.status.set("等待开始"))
        ttk.Label(settings, text="上传音频格式").grid(row=7, column=0, sticky="w", padx=(0, 16), pady=(8, 0))
        self.audio_format_selector = ttk.Combobox(
            settings, textvariable=self.audio_format, values=[fmt.value for fmt in AudioFormat], state="readonly", width=13,
        )
        self.audio_format_selector.grid(row=7, column=1, sticky="w", pady=(8, 0))
        ttk.Label(settings, text="字幕字号").grid(row=8, column=0, sticky="w", pady=(12, 0))
        ttk.Scale(settings, from_=16, to=52, value=28, command=self.set_size).grid(
            row=8, column=1, sticky="ew", pady=(12, 0),
        )
        ttk.Label(settings, textvariable=self.size_label, width=6).grid(row=8, column=2, padx=(10, 0), pady=(12, 0))
        options = ttk.Frame(settings)
        options.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.review_selector = ttk.Checkbutton(
            options, text="跨段校对（DeepSeek 文本请求）", variable=self.review_captions,
        )
        self.review_selector.pack(side="left")
        ttk.Checkbutton(
            options, text="记录对话历史", variable=self.record_history, command=self.toggle_history,
        ).pack(side="left", padx=(14, 0))
        self.history_warning = ttk.Label(settings, style="Hint.TLabel", wraplength=405)
        self.history_warning.grid(row=10, column=0, columnspan=3, sticky="w")
        self.show_history_warning(history_error)

        toggles = ttk.Frame(content)
        toggles.pack(fill="x", pady=(2, 12))
        ttk.Checkbutton(toggles, text="显示字幕", variable=self.visible, command=self.set_visible).pack(side="left")
        ttk.Checkbutton(
            toggles, text="显示原文", variable=self.show_source, command=self.render_captions,
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(toggles, text="始终置顶", variable=self.topmost, command=self.set_topmost).pack(side="left", padx=12)
        ttk.Button(toggles, text="重置位置", command=self.overlay.position_default).pack(side="right")
        actions = ttk.Frame(content)
        actions.pack(fill="x")
        self.button = ttk.Button(actions, text="开始翻译", style="Primary.TButton", command=self.toggle_session)
        self.button.pack(side="left", fill="x", expand=True)
        ttk.Button(actions, text="查看对话历史", command=self.open_history).pack(side="right", fill="y", padx=(10, 0))
        ttk.Checkbutton(
            actions, text="显示实时记录", variable=self.show_live_history, command=self.set_live_history_visible,
        ).pack(side="right", padx=(12, 0))
        ttk.Label(content, textvariable=self.status, style="Hint.TLabel", wraplength=405).pack(anchor="w", pady=(11, 0))
        self.warning = ttk.Label(content, style="Hint.TLabel", wraplength=405)
        self.warning.pack(anchor="w", pady=(3, 0))
        self.set_mode()
        self.set_scene()
        self.set_visible()
        self.mode_selector.focus_set()
        root.after(40, self.poll)
        self.schedule_document_sync()

    def set_size(self, value):
        size = round(float(value))
        self.size_label.set(f"{size} pt")
        self.overlay.set_size(size)

    def set_visible(self):
        self.overlay.set_visible(self.visible.get())

    def set_topmost(self):
        topmost = self.topmost.get()
        self.overlay.set_topmost(topmost)
        if self.live_history_window is not None:
            self.live_history_window.window.attributes("-topmost", topmost)

    def set_live_history_visible(self):
        if not self.show_live_history.get():
            if self.live_history_window is not None:
                self.close_live_history()
            return
        if self.live_history_window is None:
            self.live_history_window = LiveHistoryWindow(self.root, self.close_live_history)
            self.live_history_window.window.attributes("-topmost", self.topmost.get())
            self.live_history_window.log.update_entries(self.live_entries.values())
        else:
            window = self.live_history_window.window
            window.deiconify()
            window.lift()
        self.live_history_window.refresh_message(self.session is not None)

    def close_live_history(self):
        self.show_live_history.set(False)
        self.live_history_window.window.destroy()
        self.live_history_window = None

    def show_history_warning(self, message):
        self.history_error = message
        self.refresh_history_warning()

    def refresh_history_warning(self):
        message = self.history_error or self.document_error
        if message:
            self.history_warning.configure(text=message)
            self.history_warning.grid()
        else:
            self.history_warning.grid_remove()

    def toggle_history(self):
        enabled = self.record_history.get()
        try:
            self.history.set_enabled(enabled)
        except (OSError, sqlite3.Error):
            self.record_history.set(False)
            message = "无法开启对话历史，请检查本地数据目录。" if enabled else "本次已停止记录，无法保存开关状态。"
            self.show_history_warning(message)
        else:
            self.show_history_warning("")
            if enabled and self.session is not None:
                self.sync_history_documents()
        if self.history_window is not None:
            self.history_window.refresh_message()

    def open_history(self):
        if self.history_window is None:
            self.history_window = HistoryWindow(self.root, self.history, self.record_history, self.close_history)
        else:
            window = self.history_window.window
            window.deiconify()
            window.lift()

    def close_history(self):
        self.history_window.window.destroy()
        self.history_window = None

    def set_scene(self, _event=None):
        interview = bool(SCENE_SKILLS[self.scene.get()].questions)
        self.overlay.set_interview(interview)
        self.root.minsize(460, 730 if interview else 690)
        if interview:
            self.microphone_settings.grid()
        else:
            self.microphone_settings.grid_remove()
        self.set_microphone_state()

    def set_microphone_state(self):
        idle = self.session is None
        self.microphone_toggle.configure(state="normal" if idle else "disabled")
        self.microphone_selector.configure(state="readonly" if idle and self.capture_mic.get() else "disabled")

    def set_mode(self, event=None):
        mode = TranslationMode(self.mode.get())
        local = mode is TranslationMode.LOCAL
        self.key_entry.configure(state="normal")
        self.translation_device.configure(state="readonly" if local else "disabled")
        self.source_language.configure(state="disabled" if local else "readonly")
        if local:
            self.source.set("英语")
        if event is not None:
            self.status.set("等待开始")

    def refresh_playback_devices(self):
        self.playback_devices = self.refresh_audio_devices(
            self.playback_selector, self.playback_devices, list_playback_devices, DEFAULT_PLAYBACK, "播放设备",
        )

    def refresh_microphones(self):
        self.microphone_devices = self.refresh_audio_devices(
            self.microphone_selector, self.microphone_devices, list_microphones, DEFAULT_MICROPHONE, "麦克风",
        )

    def refresh_audio_devices(self, selector, previous_devices, list_devices, default_label, device_type):
        selected = selector.current()
        previous = previous_devices[selected - 1] if selected > 0 else None
        try:
            devices = list_devices()
        except OSError:
            self.status.set(f"无法读取 Windows {device_type}，请检查连接后重新打开列表。")
            return previous_devices

        selector.configure(values=[default_label, *(device.label for device in devices)])
        if selected == 0:
            selector.current(0)
        elif previous in devices:
            selector.current(devices.index(previous) + 1)
        else:
            selector.set("")

        if not devices:
            self.status.set(f"未找到可采集的{device_type}，请检查连接后重新打开列表。")
        elif previous is not None and previous not in devices:
            self.status.set(f"{device_type}「{previous.label}」已不可用，请重新选择。")
        return devices

    def render_captions(self):
        sources = self.transcripts if self.show_source.get() else {}
        segments = sorted(
            self.captions.keys() | sources.keys(),
            key=lambda segment: caption_order(sources.get(segment) or self.captions[segment]),
        )
        captions = [
            (sources[segment].text if segment in sources else "", self.captions.get(segment))
            for segment in segments[-2:]
        ]
        placeholder = IDLE_CAPTION
        if self.session is not None:
            placeholder = "等待语音…" if self.session.capture_mic else "等待播放语音…"
        self.overlay.render(captions, placeholder)

    def toggle_session(self):
        if self.session is not None:
            self.stopping = True
            self.session.stop()
            self.status.set("正在停止…")
            self.button.configure(state="disabled")
            return
        selected = self.playback_selector.current()
        if selected < 0:
            self.status.set("请先选择播放设备。")
            return
        playback_device = self.playback_devices[selected - 1] if selected > 0 else None
        mode = TranslationMode(self.mode.get())
        key = self.api_key.get().strip()
        if error := api_key_error(key):
            self.status.set(error)
            self.key_entry.focus_set()
            return
        skill = SCENE_SKILLS[self.scene.get()]
        capture_mic = bool(skill.questions) and self.capture_mic.get()
        microphone_device = None
        if capture_mic:
            selected = self.microphone_selector.current()
            if selected < 0:
                self.status.set("请先选择麦克风。")
                return
            if selected > 0:
                microphone_device = self.microphone_devices[selected - 1]
        self.captions.clear()
        self.transcripts.clear()
        self.live_entries.clear()
        if self.live_history_window is not None:
            self.live_history_window.log.clear()
        self.history.start_session()
        self.overlay.reset_questions(WAITING_FOR_QUESTIONS)
        self.status.set("正在启动…")
        self.warning.configure(text="")
        self.session = Interpreter(
            key, SOURCE_LANGUAGES[self.source.get()], TARGET_LANGUAGES[self.target.get()], DEVICES[self.device.get()],
            skill=skill,
            playback_device=playback_device,
            mode=mode,
            audio_format=AudioFormat(self.audio_format.get()),
            review_captions=self.review_captions.get(),
            capture_mic=capture_mic,
            microphone_device=microphone_device,
        )
        if self.show_live_history.get():
            self.set_live_history_visible()
        if self.record_history.get():
            self.sync_history_documents()
        self.render_captions()
        self.key_entry.configure(state="disabled")
        self.mode_selector.configure(state="disabled")
        self.scene_selector.configure(state="disabled")
        self.source_language.configure(state="disabled")
        self.language.configure(state="disabled")
        self.translation_device.configure(state="disabled")
        self.playback_selector.configure(state="disabled")
        self.set_microphone_state()
        self.audio_format_selector.configure(state="disabled")
        self.review_selector.configure(state="disabled")
        self.button.configure(text="停止翻译")
        self.session.start()

    def poll(self):
        redraw = False
        session_ended = False
        caption_updates = []
        if self.session is not None:
            while True:
                try:
                    event = self.session.events.get_nowait()
                except Empty:
                    break
                if event is None:
                    self.session = None
                    session_ended = True
                    if self.stopping:
                        self.status.set("已停止")
                    self.stopping = False
                    self.mode_selector.configure(state="readonly")
                    self.set_mode()
                    self.scene_selector.configure(state="readonly")
                    self.language.configure(state="readonly")
                    self.playback_selector.configure(state="readonly")
                    self.set_microphone_state()
                    self.audio_format_selector.configure(state="readonly")
                    self.review_selector.configure(state="normal")
                    self.button.configure(text="开始翻译", state="normal")
                    self.warning.configure(text="")
                    redraw = True
                    break
                is_source = isinstance(event, Transcript)
                is_caption = is_source or isinstance(event, Subtitle)
                if is_caption:
                    caption_updates.append((event, is_source))
                if self.stopping:
                    continue
                if is_caption:
                    lines = self.transcripts if is_source else self.captions
                    lines[event.segment] = event
                    for older in sorted(lines.values(), key=caption_order)[:-2]:
                        del lines[older.segment]
                    redraw = redraw or not is_source or self.show_source.get()
                elif isinstance(event, Status):
                    if event.translation:
                        self.warning.configure(text=event.text)
                    else:
                        self.status.set(event.text)
                elif isinstance(event, InterviewUpdate):
                    self.overlay.show_questions(event)
        if redraw:
            self.render_captions()
        if caption_updates:
            self.update_live_history(caption_updates)
            if self.record_history.get():
                self.save_history(caption_updates)
        if session_ended:
            if self.live_history_window is not None:
                self.live_history_window.refresh_message(False)
            self.sync_history_documents()
        self.root.after(40, self.poll)

    def update_live_history(self, updates):
        changed = {}
        for event, is_source in updates:
            entry = self.live_entries.get(event.segment)
            if entry is None:
                entry = HistoryEntry(0, event.segment, event.recorded_at or time(), "", False, "", False, event.speaker)
            if is_source and (event.final or not entry.source_final):
                entry = replace(entry, source=event.text, source_final=event.final)
            elif not is_source and (event.final or not entry.translation_final):
                entry = replace(entry, translation=event.text, translation_final=event.final)
            else:
                continue
            self.live_entries[event.segment] = entry
            changed[event.segment] = entry
        if self.live_history_window is not None:
            self.live_history_window.log.update_entries(changed.values())
            self.live_history_window.refresh_message(self.session is not None)

    def save_history(self, updates):
        try:
            entries = self.history.record(updates)
        except (OSError, sqlite3.Error):
            self.show_history_warning("无法保存对话历史，请检查本地数据目录。")
            return
        self.show_history_warning("")
        self.schedule_document_sync()
        if self.history_window is not None:
            self.history_window.update_entries(entries)

    def schedule_document_sync(self, delay=250):
        if self.document_sync_job is None:
            def sync():
                self.document_sync_job = None
                self.sync_history_documents()

            self.document_sync_job = self.root.after(delay, sync)

    def sync_history_documents(self, *, retry=True):
        try:
            if self.session is not None and self.record_history.get():
                self.history.prepare_document()
            self.history.sync_documents()
        except (OSError, sqlite3.Error):
            self.document_error = "对话文档暂时无法更新，稍后会重试。"
            if retry:
                self.schedule_document_sync(delay=2000)
        else:
            self.document_error = ""
        self.refresh_history_warning()

    def close(self):
        if self.session is not None:
            self.stopping = True
            self.session.stop()
            self.poll()
        if self.document_sync_job is not None:
            self.root.after_cancel(self.document_sync_job)
        self.sync_history_documents(retry=False)
        self.history.close()
        self.root.destroy()


if __name__ == "__main__":
    if os.name != "nt":
        raise SystemExit("AITranslate 需要 Windows 的 WASAPI 系统音频采集。")
    root = tk.Tk()
    App(root)
    root.mainloop()
