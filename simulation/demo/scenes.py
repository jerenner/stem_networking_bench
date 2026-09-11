"""Manim scenes for the 200 keV DOEELS application-workflow demonstration."""

from __future__ import annotations

import json
import os
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    GrowArrow,
    ImageMobject,
    LaggedStart,
    Line,
    MoveAlongPath,
    Rectangle,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    config,
)

BACKGROUND = "#07131b"
PANEL = "#102630"
PANEL_ALT = "#16323d"
CREAM = "#f5f0e8"
MUTED = "#9bb0ba"
CYAN = "#35d0ba"
ORANGE = "#ffb000"
BLUE = "#4da3ff"
RED = "#ef6262"
GREEN = "#41d764"
YELLOW = "#ffdc28"
TEXT_FONT = os.environ.get("LMTO_DEMO_FONT", "Avenir Next")

config.background_color = BACKGROUND
# One explicit font plus ASCII labels avoids the mixed-font fallback that
# caused uneven spacing around arrows and Unicode subscripts in the first cut.
Text.set_default(font=TEXT_FONT, disable_ligatures=False)


def _asset_directory() -> Path:
    configured = os.environ.get("LMTO_DEMO_ASSETS")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "assets" / "generated_200keV"


class DemoScene(Scene):
    """Shared visual system and independently renderable story segments."""

    def setup(self) -> None:
        super().setup()
        self.assets = _asset_directory()
        metadata_path = self.assets / "demo_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Demo assets are missing. Run demo/render_demo.sh assets first: {metadata_path}"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def title(self, text: str, kicker: str | None = None) -> VGroup:
        heading = Text(text, color=CREAM, font_size=37, weight="BOLD")
        if heading.width > 13.0:
            heading.scale_to_fit_width(13.0)
        if kicker is None:
            return VGroup(heading).to_edge(UP, buff=0.3)
        overline = Text(kicker.upper(), color=CYAN, font_size=15, weight="BOLD")
        return (
            VGroup(overline, heading)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            .to_edge(UP, buff=0.2)
            .to_edge(LEFT, buff=0.48)
        )

    def caption(self, text: str, color: str = CYAN) -> VGroup:
        label = Text(text, color=CREAM, font_size=19)
        if label.width > 12.2:
            label.scale_to_fit_width(12.2)
        panel = RoundedRectangle(
            width=max(label.width + 0.55, 4.0),
            height=0.58,
            corner_radius=0.1,
            fill_color=PANEL,
            fill_opacity=0.97,
            stroke_color=color,
            stroke_width=1.0,
        )
        label.move_to(panel)
        return VGroup(panel, label).to_edge(DOWN, buff=0.2)

    def chip(self, value: str, label: str, color: str = CYAN, width: float = 2.2) -> VGroup:
        number = Text(value, color=color, font_size=25, weight="BOLD")
        description = Text(label, color=MUTED, font_size=12)
        contents = VGroup(number, description).arrange(DOWN, buff=0.03)
        box = RoundedRectangle(
            width=max(width, contents.width + 0.28),
            height=0.82,
            corner_radius=0.1,
            fill_color=PANEL,
            fill_opacity=1.0,
            stroke_color="#35505d",
            stroke_width=1.0,
        )
        contents.move_to(box)
        return VGroup(box, contents)

    def image_card(self, filename: str, width: float = 10.4, shift=ORIGIN) -> Group:
        image = ImageMobject(str(self.assets / filename)).set_width(width).shift(shift)
        frame = RoundedRectangle(
            width=image.width + 0.12,
            height=image.height + 0.12,
            corner_radius=0.08,
            fill_opacity=0,
            stroke_color="#35505d",
            stroke_width=1.1,
        ).move_to(image)
        return Group(frame, image)

    def body_lines(
        self,
        lines: list[str],
        colors: list[str] | None = None,
        font_size: int = 18,
        max_width: float = 6.0,
    ) -> VGroup:
        colors = colors or [CREAM] * len(lines)
        items = VGroup()
        for line, color in zip(lines, colors):
            label = Text("-  " + line, color=color, font_size=font_size)
            if label.width > max_width:
                label.scale_to_fit_width(max_width)
            items.add(label)
        return items.arrange(DOWN, aligned_edge=LEFT, buff=0.2)

    def stage_box(self, label: str, detail: str, color: str, width: float = 2.5) -> VGroup:
        heading = Text(label, color=color, font_size=17, weight="BOLD")
        note = Text(detail, color=MUTED, font_size=11)
        contents = VGroup(heading, note).arrange(DOWN, buff=0.08)
        box = RoundedRectangle(
            width=width,
            height=1.15,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=1.0,
            stroke_color=color,
            stroke_width=1.4,
        )
        contents.move_to(box)
        return VGroup(box, contents)

    def clear_scene(self, run_time: float = 0.35) -> None:
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=run_time)

    def intro_segment(self) -> None:
        eyebrow = Text("DOEELS + GPU-NATIVE ACQUISITION", color=CYAN, font_size=18, weight="BOLD")
        title = Text(
            "Structure and chemistry\nin the same scan",
            color=CREAM,
            font_size=49,
            weight="BOLD",
            line_spacing=0.88,
        )
        subtitle = Text(
            "An LMTO simulation for chip and battery workflows",
            color=MUTED,
            font_size=19,
        )
        copy = (
            VGroup(eyebrow, title, subtitle)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            .to_edge(LEFT, buff=0.72)
        )
        accent = Rectangle(
            width=0.1, height=2.9, fill_color=ORANGE, fill_opacity=1, stroke_width=0
        ).next_to(copy, LEFT, buff=0.23)
        cards = (
            Group(
                *[
                    self.image_card(name, width=1.55)
                    for name in (
                        "final_haadf.png",
                        "final_mn.png",
                        "final_o.png",
                        "final_ti.png",
                    )
                ]
            )
            .arrange_in_grid(rows=2, cols=2, buff=0.12)
            .move_to(RIGHT * 4.7)
        )
        self.play(
            FadeIn(accent, shift=UP * 0.15),
            FadeIn(copy, shift=UP * 0.15),
            FadeIn(cards, shift=LEFT * 0.2),
            run_time=1.0,
        )
        self.wait(2.7)
        self.clear_scene()

    def application_segment(self) -> None:
        title = self.title("The end application: find where each element is", "01 / workflow goal")
        card = self.image_card("application_workflows.png", width=11.7, shift=DOWN * 0.05)
        caption = self.caption(
            "Choose a device, particle, or interface; scan it once; inspect structure and composition."
        )
        self.play(FadeIn(title), FadeIn(card), run_time=0.9)
        self.play(FadeIn(caption))
        self.wait(4.0)
        self.clear_scene()

    def views_segment(self) -> None:
        title = self.title("One scan produces several scientific views", "02 / result")
        names = ["Structure", "Mn", "O", "Ti"]
        files = ["final_haadf.png", "final_mn.png", "final_o.png", "final_ti.png"]
        colors = [CREAM, GREEN, YELLOW, RED]
        panels = Group()
        for name, filename, color in zip(names, files, colors):
            image = self.image_card(filename, width=2.65)
            label = Text(name, color=color, font_size=20, weight="BOLD").next_to(
                image, UP, buff=0.1
            )
            panels.add(Group(image, label))
        panels.arrange(RIGHT, buff=0.22).move_to(DOWN * 0.05)
        caption = self.caption(
            "HAADF shows the larger structure; energy-loss signatures select individual elements."
        )
        self.play(
            FadeIn(title),
            LaggedStart(*[FadeIn(panel) for panel in panels], lag_ratio=0.12),
            run_time=1.0,
        )
        self.play(FadeIn(caption))
        self.wait(3.5)
        self.clear_scene()

    def specimen_segment(self) -> None:
        title = self.title("Demonstration material: an LMTO lattice", "03 / specimen")
        structure = self.image_card("structure_3d.png", width=7.2, shift=LEFT * 3.0 + DOWN * 0.08)
        specimen = self.metadata["specimen"]
        chips = (
            VGroup(
                self.chip("Li1.2Mn0.4Ti0.4O2", "battery cathode composition", YELLOW, 3.45),
                self.chip(
                    str(specimen["atom_count"]),
                    "atoms in the periodic cell",
                    CYAN,
                    3.45,
                ),
                self.chip("0.83 x 0.83 x 1.66 nm", "simulated volume", BLUE, 3.45),
                self.chip("40 Li / 16 Mn / 8 Ti / 64 O", "random-cation rocksalt", GREEN, 3.45),
            )
            .arrange(DOWN, buff=0.17)
            .move_to(RIGHT * 4.45 + DOWN * 0.05)
        )
        self.play(FadeIn(title), FadeIn(structure), FadeIn(chips), run_time=0.9)
        self.wait(3.6)
        planes = self.image_card("structure_planes.png", width=10.7, shift=DOWN * 0.05)
        caption = self.caption(
            "All eight atomic planes are explicit; the cell is synthetic, periodic, and unrelaxed.",
            ORANGE,
        )
        self.play(FadeOut(structure), FadeOut(chips), FadeIn(planes), run_time=0.75)
        self.play(FadeIn(caption))
        self.wait(3.4)
        self.clear_scene()

    def scan_segment(self) -> None:
        title = self.title("A focused 200 keV probe steps across the lattice", "04 / scan")
        raster = self.image_card("probe_raster.png", width=7.0, shift=LEFT * 2.85 + DOWN * 0.08)
        scan = self.metadata["scan"]
        chips = (
            VGroup(
                self.chip("16 x 16", "probe positions", CYAN, 2.85),
                self.chip(f"{scan['step_A']:.3f} Angstrom", "step size", BLUE, 2.85),
                self.chip(f"{scan['beam_current_pA']:.0f} pA", "beam current", ORANGE, 2.85),
                self.chip(f"{scan['dwell_ms']:.1f} ms", "per position", GREEN, 2.85),
                self.chip(
                    f"{scan['electrons_per_position'] / 1e6:.3f} million",
                    "electrons per position",
                    YELLOW,
                    2.85,
                ),
            )
            .arrange(DOWN, buff=0.12)
            .move_to(RIGHT * 4.55 + DOWN * 0.05)
        )
        caption = self.caption(
            "At every point we collect structural contrast and a complete energy-loss readout."
        )
        self.play(FadeIn(title), FadeIn(raster), FadeIn(chips), run_time=0.9)
        self.play(FadeIn(caption))
        self.wait(3.8)
        self.clear_scene()

    def measurement_segment(self) -> None:
        title = self.title("Two measurements are made at each position", "05 / microscope")
        gun = Circle(radius=0.5, color=CYAN, fill_color=PANEL_ALT, fill_opacity=1).move_to(
            LEFT * 5.8
        )
        gun_label = Text("200 keV\ne- beam", color=CREAM, font_size=16, line_spacing=0.84).move_to(
            gun
        )
        sample = Rectangle(
            width=0.42,
            height=2.65,
            fill_color=YELLOW,
            fill_opacity=0.72,
            stroke_color=ORANGE,
        ).move_to(LEFT * 3.35)
        sample_label = (
            Text("LMTO", color=CREAM, font_size=15, weight="BOLD").rotate(1.5708).move_to(sample)
        )
        haadf = self.stage_box("HAADF", "structure", CREAM, 2.5).move_to(LEFT * 0.4 + UP * 1.45)
        eels = self.stage_box("DOEELS", "energy lost", ORANGE, 2.5).move_to(
            LEFT * 0.4 + DOWN * 1.05
        )
        camera = self.stage_box("FAST CAMERA", "pixelated silicon", BLUE, 2.75).move_to(
            RIGHT * 4.0 + DOWN * 1.05
        )
        beam = Arrow(gun.get_right(), sample.get_left(), buff=0.08, color=CYAN, stroke_width=4)
        branch_1 = Arrow(
            sample.get_right(), haadf.get_left(), buff=0.1, color=CREAM, stroke_width=3
        )
        branch_2 = Arrow(
            sample.get_right(), eels.get_left(), buff=0.1, color=ORANGE, stroke_width=3
        )
        to_camera = Arrow(eels.get_right(), camera.get_left(), buff=0.1, color=BLUE, stroke_width=3)
        notes = (
            VGroup(
                Text("how strongly electrons scatter", color=MUTED, font_size=15),
                Text("how much energy electrons lose", color=MUTED, font_size=15),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=1.55)
            .move_to(RIGHT * 4.0 + UP * 1.15)
        )
        self.play(
            FadeIn(title),
            FadeIn(gun),
            FadeIn(gun_label),
            GrowArrow(beam),
            FadeIn(sample),
            FadeIn(sample_label),
            run_time=0.8,
        )
        self.play(
            GrowArrow(branch_1),
            FadeIn(haadf),
            GrowArrow(branch_2),
            FadeIn(eels),
            GrowArrow(to_camera),
            FadeIn(camera),
            FadeIn(notes),
            run_time=1.0,
        )
        caption = self.caption(
            "HAADF and DOEELS are synchronized branches of the same raster scan."
        )
        self.play(FadeIn(caption))
        self.wait(3.6)
        self.clear_scene()

    def raw_segment(self) -> None:
        title = self.title("Raw camera frames are only the starting point", "06 / detector data")
        raw = self.image_card("raw_detector_frame.png", width=11.6, shift=DOWN * 0.03)
        caption = self.caption(
            "One readout is a noisy 960 x 3840 pixel image - not an element map."
        )
        self.play(FadeIn(title), FadeIn(raw), run_time=0.8)
        # The raster image is deliberately large; keep the title above it
        # throughout the cross-fade into the reduction diagram.
        self.bring_to_front(title)
        self.play(FadeIn(caption))
        self.wait(3.2)
        stages = (
            VGroup(
                self.stage_box("CORRECT", "pedestal + noise", BLUE, 2.25),
                self.stage_box("REDUCE", "sum detector rows", CYAN, 2.25),
                self.stage_box("FOLD", "repeat ZLP lanes", ORANGE, 2.25),
                self.stage_box("ACCUMULATE", "many short frames", GREEN, 2.45),
                self.stage_box("SPECTRUM", "counts vs energy", YELLOW, 2.25),
            )
            .arrange(RIGHT, buff=0.28)
            .move_to(UP * 0.25)
        )
        arrows = VGroup(
            *[
                Arrow(
                    stages[i].get_right(),
                    stages[i + 1].get_left(),
                    buff=0.04,
                    color="#58727d",
                    stroke_width=2,
                )
                for i in range(len(stages) - 1)
            ]
        )
        explanation = Text(
            "The important output is a calibrated energy histogram for each probe position.",
            color=CREAM,
            font_size=22,
        ).move_to(DOWN * 1.45)
        self.play(
            FadeOut(raw),
            FadeOut(caption),
            FadeIn(stages),
            Create(arrows),
            FadeIn(explanation),
            run_time=0.9,
        )
        self.wait(3.0)
        self.clear_scene()

    def spectrum_segment(self) -> None:
        title = self.title("A spectrum tells us how much energy was lost", "07 / reconstruction")
        card = self.image_card("analog_vs_counted_processing.png", width=11.7, shift=DOWN * 0.03)
        self.play(FadeIn(title), FadeIn(card), run_time=0.9)
        fitted = self.metadata["spectrum_example"]["fitted_edge_counts"]
        caption = self.caption(
            f"At this Mn-rich point the fit assigns about {fitted['Mn L₂,₃']:.0f} counts to Mn; known edge energies identify each element."
        )
        self.play(FadeIn(caption))
        self.wait(4.7)
        self.clear_scene()

    def maps_segment(self) -> None:
        title = self.title("Repeat the spectrum measurement to build a map", "08 / element mapping")
        raster = self.image_card("probe_raster.png", width=4.4, shift=LEFT * 4.55 + DOWN * 0.08)
        fit = self.stage_box("FIT EDGES", "456 / 532 / 640 eV", ORANGE, 2.25).move_to(
            LEFT * 0.55 + DOWN * 0.05
        )
        maps = (
            Group(
                self.image_card("final_mn.png", width=1.85),
                self.image_card("final_o.png", width=1.85),
                self.image_card("final_ti.png", width=1.85),
            )
            .arrange(RIGHT, buff=0.1)
            .move_to(RIGHT * 4.0 + DOWN * 0.05)
        )
        labels = VGroup(
            Text("Mn", color=GREEN, font_size=16, weight="BOLD"),
            Text("O", color=YELLOW, font_size=16, weight="BOLD"),
            Text("Ti", color=RED, font_size=16, weight="BOLD"),
        )
        for label, image in zip(labels, maps):
            label.next_to(image, UP, buff=0.08)
        arrow_1 = Arrow(raster.get_right(), fit.get_left(), buff=0.1, color=CYAN, stroke_width=3)
        arrow_2 = Arrow(fit.get_right(), maps.get_left(), buff=0.1, color=ORANGE, stroke_width=3)
        self.play(
            FadeIn(title),
            FadeIn(raster),
            GrowArrow(arrow_1),
            FadeIn(fit),
            GrowArrow(arrow_2),
            FadeIn(maps),
            FadeIn(labels),
            run_time=1.2,
        )
        caption = self.caption(
            "One fitted edge amplitude at every (x, y) position becomes one elemental map."
        )
        self.play(FadeIn(caption))
        self.wait(3.7)
        self.clear_scene()

    def daqiri_segment(self) -> None:
        title = self.title("The data path must keep up with the camera", "09 / GPU-native DAQ")
        stages = (
            VGroup(
                self.stage_box("CAMERA", "short readouts", BLUE, 2.35),
                self.stage_box("DAQIRI", "network -> GPU", CYAN, 2.35),
                self.stage_box("GPU", "correct + count", GREEN, 2.35),
                self.stage_box("LIVE OUTPUT", "spectra + maps", YELLOW, 2.45),
            )
            .arrange(RIGHT, buff=0.62)
            .move_to(UP * 0.45)
        )
        paths = VGroup(
            *[
                Line(
                    stages[i].get_right(),
                    stages[i + 1].get_left(),
                    color="#58727d",
                    stroke_width=4,
                )
                for i in range(len(stages) - 1)
            ]
        )
        self.play(FadeIn(title), FadeIn(stages), Create(paths), run_time=0.9)
        dots = VGroup()
        animations = []
        for index in range(9):
            path = paths[index % len(paths)]
            dot = Dot(path.get_start(), radius=0.055, color=(BLUE, CYAN, GREEN)[index % 3])
            dots.add(dot)
            animations.append(MoveAlongPath(dot, path, run_time=0.7))
        self.add(dots)
        self.play(LaggedStart(*animations, lag_ratio=0.12), run_time=2.0)
        storage = self.stage_box("SELECTIVE HDF5", "bursts when needed", ORANGE, 2.65).move_to(
            RIGHT * 2.1 + DOWN * 1.5
        )
        storage_arrow = Arrow(
            stages[2].get_bottom(),
            storage.get_top(),
            buff=0.12,
            color=ORANGE,
            stroke_width=2.5,
        )
        note = Text("high-rate data stay on the GPU path", color=CREAM, font_size=22).move_to(
            LEFT * 2.45 + DOWN * 1.5
        )
        self.play(FadeIn(note), GrowArrow(storage_arrow), FadeIn(storage), run_time=0.8)
        caption = self.caption(
            "Advanced cameras create the data; DAQIRI makes streaming analysis practical."
        )
        self.play(FadeIn(caption))
        self.wait(3.4)
        self.clear_scene()

    def value_segment(self) -> None:
        title = self.title("The improvement is time-to-map, not only data rate", "10 / capability")
        old = (
            VGroup(
                self.stage_box("ACQUIRE", "finish scan", MUTED, 2.0),
                self.stage_box("WRITE", "large raw set", MUTED, 2.0),
                self.stage_box("MOVE", "another system", MUTED, 2.0),
                self.stage_box("ANALYZE", "afterward", MUTED, 2.0),
                self.stage_box("MAP", "later", MUTED, 2.0),
            )
            .arrange(RIGHT, buff=0.22)
            .move_to(UP * 1.15)
        )
        old_label = Text("OFFLINE", color=MUTED, font_size=16, weight="BOLD").next_to(
            old, LEFT, buff=0.2
        )
        new = (
            VGroup(
                self.stage_box("ACQUIRE", "continuous", BLUE, 2.25),
                self.stage_box("GPU PIPELINE", "receive + correct", CYAN, 2.55),
                self.stage_box("ACCUMULATE", "spectrum at each point", GREEN, 2.65),
                self.stage_box("MAP", "updates during scan", YELLOW, 2.4),
            )
            .arrange(RIGHT, buff=0.32)
            .move_to(DOWN * 0.45)
        )
        new_label = Text("STREAMING", color=CYAN, font_size=16, weight="BOLD").next_to(
            new, LEFT, buff=0.2
        )
        arrows_old = VGroup(
            *[
                Arrow(
                    old[i].get_right(),
                    old[i + 1].get_left(),
                    buff=0.03,
                    color=MUTED,
                    stroke_width=1.7,
                )
                for i in range(len(old) - 1)
            ]
        )
        arrows_new = VGroup(
            *[
                Arrow(
                    new[i].get_right(),
                    new[i + 1].get_left(),
                    buff=0.03,
                    color=CYAN,
                    stroke_width=2.2,
                )
                for i in range(len(new) - 1)
            ]
        )
        self.play(
            FadeIn(title),
            FadeIn(old),
            FadeIn(old_label),
            Create(arrows_old),
            run_time=0.9,
        )
        self.play(FadeIn(new), FadeIn(new_label), Create(arrows_new), run_time=0.9)
        caption = self.caption(
            "Less data movement and earlier feedback; matched end-to-end speedup is still to be measured.",
            ORANGE,
        )
        self.play(FadeIn(caption))
        self.wait(4.0)
        self.clear_scene()

    def dose_segment(self) -> None:
        title = self.title(
            "Streaming accumulation shows when the map is usable", "11 / dose feedback"
        )
        card = self.image_card("exposure_sweep_correlations.png", width=10.9, shift=UP * 0.05)
        correlations = self.metadata["final_correlations"]
        chips = (
            VGroup(
                self.chip(f"{correlations['Mn']:.2f}", "Mn map correlation", GREEN, 2.25),
                self.chip(f"{correlations['O']:.2f}", "O map correlation", YELLOW, 2.25),
                self.chip(f"{correlations['Ti']:.2f}", "Ti map correlation", RED, 2.25),
            )
            .arrange(RIGHT, buff=0.2)
            .scale(0.78)
            .to_edge(DOWN, buff=0.78)
        )
        self.play(FadeIn(title), FadeIn(card), run_time=0.8)
        self.play(FadeIn(chips), run_time=0.5)
        caption = self.caption(
            "In this model, Mn, O, and Ti maps are recognizable near 35 ms per position."
        )
        self.play(FadeIn(caption))
        self.wait(3.6)
        self.clear_scene()

    def result_segment(self) -> None:
        title = self.title("DOEELS allows selection by element", "12 / element selection")
        channel_names = ["Structure", "Mn", "O", "Ti"]
        filenames = ["final_haadf.png", "final_mn.png", "final_o.png", "final_ti.png"]
        colors = [CREAM, GREEN, YELLOW, RED]
        labels = (
            VGroup(
                *[
                    Text(name, color=color, font_size=15, weight="BOLD")
                    for name, color in zip(channel_names, colors)
                ]
            )
            .arrange(RIGHT, buff=1.05)
            .to_edge(DOWN, buff=0.83)
        )
        rail = Line(
            labels[0].get_center() + UP * 0.36,
            labels[-1].get_center() + UP * 0.36,
            color="#526d78",
            stroke_width=4,
        )
        ticks = VGroup(
            *[
                Line(
                    label.get_center() + UP * 0.26,
                    label.get_center() + UP * 0.47,
                    color=color,
                    stroke_width=2,
                )
                for label, color in zip(labels, colors)
            ]
        )
        knob = Dot(labels[0].get_center() + UP * 0.36, radius=0.11, color=colors[0])
        current = self.image_card(filenames[0], width=5.8, shift=UP * 0.15)
        descriptions = [
            "High-angle\nstructural contrast",
            "Mn signature\nstarts near 640 eV",
            "O signature\nstarts near 532 eV",
            "Ti signature\nstarts near 456 eV",
        ]
        description = Text(descriptions[0], color=MUTED, font_size=17, line_spacing=0.9).move_to(
            RIGHT * 4.75 + UP * 0.25
        )
        self.play(
            FadeIn(title),
            FadeIn(current),
            Create(rail),
            FadeIn(ticks),
            FadeIn(labels),
            FadeIn(knob),
            FadeIn(description),
            run_time=1.0,
        )
        for index in range(1, len(channel_names)):
            next_card = self.image_card(filenames[index], width=5.8, shift=UP * 0.15)
            next_description = Text(
                descriptions[index], color=colors[index], font_size=17, line_spacing=0.9
            ).move_to(RIGHT * 4.75 + UP * 0.25)
            self.play(
                knob.animate.move_to(labels[index].get_center() + UP * 0.36).set_color(
                    colors[index]
                ),
                FadeOut(current),
                FadeIn(next_card),
                FadeOut(description),
                FadeIn(next_description),
                run_time=0.8,
            )
            current = next_card
            description = next_description
            self.wait(1.25)
        caption = self.caption("Every channel comes from the same 16 x 16 acquisition.").to_edge(
            DOWN, buff=0.02
        )
        self.play(FadeIn(caption))
        self.wait(2.5)
        self.clear_scene()

    def workflow_return_segment(self) -> None:
        title = self.title("The same output scales to practical samples", "13 / end workflow")
        card = self.image_card("application_workflows.png", width=11.7, shift=DOWN * 0.05)
        self.play(FadeIn(title), FadeIn(card), run_time=0.9)
        caption = self.caption(
            "Inspect a transistor layer or a battery interface by switching the selected element."
        )
        self.play(FadeIn(caption))
        self.wait(3.8)
        self.clear_scene()

    def provenance_segment(self) -> None:
        title = self.title("A physics-based demonstration, with clear limits", "14 / provenance")
        left_box = RoundedRectangle(
            width=5.9,
            height=4.65,
            corner_radius=0.16,
            fill_color=PANEL,
            fill_opacity=0.92,
            stroke_color=CYAN,
            stroke_width=2,
        ).move_to(LEFT * 3.2)
        right_box = RoundedRectangle(
            width=5.9,
            height=4.65,
            corner_radius=0.16,
            fill_color=PANEL,
            fill_opacity=0.92,
            stroke_color=ORANGE,
            stroke_width=2,
        ).move_to(RIGHT * 3.2)
        left_title = Text("WHAT IS MODELED", color=CYAN, font_size=20, weight="BOLD").move_to(
            left_box.get_top() + DOWN * 0.42
        )
        right_title = Text(
            "WHAT STILL NEEDS DATA", color=ORANGE, font_size=20, weight="BOLD"
        ).move_to(right_box.get_top() + DOWN * 0.42)
        left_lines = (
            self.body_lines(
                [
                    "200 keV probe propagation through all atoms",
                    "HAADF and element-edge spatial response",
                    "energy-loss probabilities and spectra",
                    "spectrometer and silicon charge response",
                    "measured NiO pedestal and noise",
                ],
                font_size=16,
            )
            .move_to(left_box)
            .shift(DOWN * 0.15)
        )
        right_lines = (
            self.body_lines(
                [
                    "larger, relaxed real sample structures",
                    "material-specific chemical fine structure",
                    "measured spectrometer calibration",
                    "confirmed active sensor geometry",
                    "matched end-to-end timing benchmark",
                ],
                font_size=16,
            )
            .move_to(right_box)
            .shift(DOWN * 0.15)
        )
        footer = Text(
            "abTEM: spatial wave physics   |   GOSH: edge strengths   |   Geant4 + NiO: detector response",
            color=MUTED,
            font_size=15,
        ).to_edge(DOWN, buff=0.83)
        self.play(
            FadeIn(title),
            FadeIn(left_box),
            FadeIn(right_box),
            FadeIn(left_title),
            FadeIn(right_title),
            run_time=0.8,
        )
        self.play(
            LaggedStart(
                *[FadeIn(line, shift=RIGHT * 0.08) for line in left_lines],
                lag_ratio=0.1,
            ),
            LaggedStart(
                *[FadeIn(line, shift=RIGHT * 0.08) for line in right_lines],
                lag_ratio=0.1,
            ),
            FadeIn(footer),
            run_time=1.1,
        )
        caption = self.caption(
            "The element maps are generated from simulated measurements - not painted atom labels.",
            ORANGE,
        )
        self.play(FadeIn(caption))
        self.wait(3.7)
        self.clear_scene()

    def outro_segment(self) -> None:
        title = Text(
            "From fast detector data to useful chemical maps",
            color=CREAM,
            font_size=40,
            weight="BOLD",
        )
        if title.width > 13.0:
            title.scale_to_fit_width(13.0)
        flow = Text(
            "sample -> fast camera -> DAQIRI + GPU -> spectrum -> element map",
            color=CYAN,
            font_size=21,
        )
        note = Text(
            "One continuous workflow for structure, composition, and live feedback.",
            color=MUTED,
            font_size=18,
        )
        group = VGroup(title, flow, note).arrange(DOWN, buff=0.34)
        self.play(FadeIn(group, shift=UP * 0.18), run_time=0.9)
        self.wait(3.0)
        self.play(FadeOut(group), run_time=0.6)


class ApplicationScene(DemoScene):
    def construct(self):
        self.application_segment()


class SpecimenScene(DemoScene):
    def construct(self):
        self.specimen_segment()


class SpectrumScene(DemoScene):
    def construct(self):
        self.raw_segment()
        self.spectrum_segment()
        self.maps_segment()


class DaqiriScene(DemoScene):
    def construct(self):
        self.daqiri_segment()
        self.value_segment()


class ResultsScene(DemoScene):
    def construct(self):
        self.dose_segment()
        self.result_segment()


class FullDemo(DemoScene):
    def construct(self):
        self.intro_segment()
        self.application_segment()
        self.views_segment()
        self.specimen_segment()
        self.scan_segment()
        self.measurement_segment()
        self.raw_segment()
        self.spectrum_segment()
        self.maps_segment()
        self.daqiri_segment()
        self.value_segment()
        self.dose_segment()
        self.result_segment()
        self.workflow_return_segment()
        self.provenance_segment()
        self.outro_segment()
