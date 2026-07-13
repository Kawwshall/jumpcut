"""jumpcut — AI auto-editor for Loom videos.

    jumpcut <file-or-loom-url> [<file-or-url> ...] [options]

Multiple sources are stitched together, in the order given, into one timeline
before editing.

Pipeline: ingest -> (concat if multiple) -> transcribe -> analyze -> render.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import render as render_mod
from .models import CutReason
from .pipeline import PipelineOptions, has_anthropic, run_pipeline

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.command()
def edit(
    sources: list[str] = typer.Argument(
        ..., help="One or more local video paths / Loom share URLs. "
                  "Multiple clips are stitched together in the order given."
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Output .mp4 path."),
    silence: bool = typer.Option(True, help="Cut long pauses / dead air."),
    fillers: bool = typer.Option(True, help="Remove filler words (um, uh, ...)."),
    captions: bool = typer.Option(True, help="Burn captions into the video."),
    retakes: bool = typer.Option(False, help="Cut false starts/retakes (needs ANTHROPIC_API_KEY)."),
    min_gap: float = typer.Option(0.6, help="Min pause length (s) to count as silence."),
    aggressive_fillers: bool = typer.Option(
        False, help="Also cut soft fillers (like, so, actually...). Riskier."
    ),
    model: str = typer.Option("base", help="Whisper size: tiny/base/small/medium/large-v3."),
    dry_run: bool = typer.Option(False, help="Analyze and report, but don't render."),
    keep_srt: bool = typer.Option(False, help="Also write the .srt sidecar file."),
):
    """Auto-edit a Loom: cut silence + fillers, optionally retakes, burn captions."""
    workdir = Path(".jumpcut").resolve()

    if retakes and not has_anthropic():
        console.print("[yellow]![/] Retake detection needs the 'retakes' extra + "
                      "ANTHROPIC_API_KEY — skipping.")

    opts = PipelineOptions(
        silence=silence, fillers=fillers, captions=captions, retakes=retakes,
        min_gap=min_gap, aggressive_fillers=aggressive_fillers,
        whisper_model=model, keep_srt=keep_srt,
    )

    progress_box: dict = {}

    def on_stage(stage: str, **info):
        if stage == "resolved":
            n = info.get("total", 1)
            prefix = f"[{info['index'] + 1}/{n}] " if n > 1 else ""
            dur = info.get("duration", 0)
            warn = "  [yellow]⚠ looks short — check this URL/file[/]" if dur and dur < 5 else ""
            console.print(
                f"[green]✓[/] {prefix}Resolved: [bold]{info['filename']}[/] "
                f"([cyan]{dur:.1f}s[/]){warn}"
            )
        elif stage == "concatenating":
            console.print(f"[cyan]…[/] Stitching {info['count']} clips together...")
        elif stage == "concatenated":
            console.print(f"[green]✓[/] Combined into: [bold]{info['filename']}[/]")
        elif stage == "transcribed":
            console.print(f"[green]✓[/] Transcribed {info['words']} words, "
                           f"{info['duration']:.0f}s.")
        elif stage == "planned":
            _report(info["plan"])
        elif stage == "nothing_left":
            console.print("[red]Nothing left after cuts. Loosen your settings.[/]")
        elif stage == "rendered":
            mode = info["mode"]
            if mode == "burned":
                console.print("[green]✓[/] Captions burned in.")
            elif mode == "soft":
                console.print("[green]✓[/] Captions added as a soft subtitle track.")
        progress_box["stage"] = stage

    if captions and not render_mod.can_burn_captions():
        console.print(
            "[yellow]![/] This ffmpeg build is missing even the `overlay` filter — "
            "captions will be added as a [bold]soft (toggleable) track[/] instead "
            "of burned in. This is extremely unusual; consider reinstalling ffmpeg."
        )

    with Progress(
        SpinnerColumn(), TextColumn("[cyan]{task.fields[label]}"),
        console=console, transient=True,
    ) as prog:
        task = prog.add_task("p", label="Working...")

        def _live_stage(stage: str, **info):
            n = info.get("total", 1)
            labels = {
                "resolving": f"Resolving clip {info.get('index', 0) + 1}/{n}...",
                "concatenating": f"Stitching {info.get('count', 0)} clips together...",
                "transcribing": f"Transcribing ({info.get('done', 0):.0f}s / "
                                f"{info.get('total', 0):.0f}s)",
                "analyzing": "Planning edits...",
                "rendering": "Rendering with ffmpeg (re-encoding)...",
            }
            if stage in labels:
                prog.update(task, label=labels[stage])
            on_stage(stage, **info)

        try:
            result = run_pipeline(
                sources, output, opts, workdir, on_stage=_live_stage, dry_run=dry_run,
            )
        except RuntimeError as exc:
            prog.stop()
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1)

    if dry_run:
        console.print("[yellow]Dry run — nothing rendered.[/]")
        raise typer.Exit()

    if result.srt:
        console.print(f"[green]✓[/] Captions: [bold]{result.srt}[/]")
    console.print(f"\n[bold green]Done![/] → [bold]{result.output}[/]")


def _report(plan) -> None:
    counts: dict[CutReason, tuple[int, float]] = {}
    for c in plan.cuts:
        n, d = counts.get(c.reason, (0, 0.0))
        counts[c.reason] = (n + 1, d + c.duration)

    table = Table(title="Edit plan", show_header=True, header_style="bold cyan")
    table.add_column("Reason")
    table.add_column("Cuts", justify="right")
    table.add_column("Time removed", justify="right")
    for reason, (n, d) in counts.items():
        table.add_row(reason.value, str(n), f"{d:.1f}s")
    console.print(table)

    o, f = plan.original_duration, plan.final_duration
    pct = (plan.removed_duration / o * 100) if o else 0
    console.print(
        f"[bold]{o:.0f}s → {f:.0f}s[/]  "
        f"([green]-{plan.removed_duration:.0f}s, {pct:.0f}% shorter[/])"
    )


if __name__ == "__main__":
    app()
