"""Manim scenes for the reproducible STEM DAQIRI NiO demonstration."""

from __future__ import annotations

import json
import os
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    GrowArrow,
    ImageMobject,
    Indicate,
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
MUTED = "#8fa3ad"
CYAN = "#35d0ba"
ORANGE = "#ffb000"
RED = "#ef6262"
BLUE = "#4da3ff"
TEXT_FONT = os.environ.get("STEM_DEMO_FONT", "Helvetica Neue")

config.background_color = BACKGROUND
# Pango's empty default font can resolve to a display serif whose spacing is
# difficult to read at annotation sizes. Use one explicit face for stable,
# legible labels throughout the rendered video.
Text.set_default(font=TEXT_FONT)


def _asset_directory() -> Path:
    configured = os.environ.get("STEM_DEMO_ASSETS")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "assets" / "generated"


class DemoScene(Scene):
    """Shared visual system and independently callable story segments."""

    def setup(self):
        super().setup()
        self.assets = _asset_directory()
        metadata_path = self.assets / "demo_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                "Demo assets are missing. Run demo/render_demo.sh assets first: {}".format(
                    metadata_path
                )
            )
        with metadata_path.open("r", encoding="utf-8") as stream:
            self.metadata = json.load(stream)

    def title(self, text: str, kicker: str | None = None) -> VGroup:
        heading = Text(text, color=CREAM, font_size=37, weight="BOLD")
        if kicker is None:
            return VGroup(heading).to_edge(UP, buff=0.35)
        overline = Text(kicker.upper(), color=CYAN, font_size=16, weight="BOLD")
        return (
            VGroup(overline, heading)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            .to_edge(UP, buff=0.26)
            .to_edge(LEFT, buff=0.55)
        )

    def caption(self, text: str) -> VGroup:
        label = Text(text, color=CREAM, font_size=22)
        panel = RoundedRectangle(
            width=min(max(label.width + 0.7, 4.0), 13.2),
            height=0.62,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=0.96,
            stroke_color="#35505d",
            stroke_width=1.0,
        )
        label.move_to(panel)
        return VGroup(panel, label).to_edge(DOWN, buff=0.25)

    def chip(self, value: str, label: str, color: str = CYAN) -> VGroup:
        number = Text(value, color=color, font_size=28, weight="BOLD")
        description = Text(label, color=MUTED, font_size=13)
        contents = VGroup(number, description).arrange(DOWN, buff=0.04)
        box = RoundedRectangle(
            width=max(contents.width + 0.35, 2.0),
            height=0.9,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=1.0,
            stroke_color="#35505d",
            stroke_width=1,
        )
        contents.move_to(box)
        return VGroup(box, contents)

    def image_card(self, filename: str, width: float = 10.2) -> Group:
        image = ImageMobject(str(self.assets / filename)).set_width(width)
        frame = RoundedRectangle(
            width=image.width + 0.16,
            height=image.height + 0.16,
            corner_radius=0.1,
            fill_opacity=0,
            stroke_color="#35505d",
            stroke_width=1.3,
        ).move_to(image)
        return Group(frame, image)

    def clear_scene(self, run_time: float = 0.45) -> None:
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=run_time)

    def intro_segment(self) -> None:
        eyebrow = Text("STEM NETWORKING BENCH", color=CYAN, font_size=18, weight="BOLD")
        title = Text(
            "From detector packets\nto a live NiO EELS spectrum",
            color=CREAM,
            font_size=48,
            weight="BOLD",
            line_spacing=0.9,
        )
        subtitle = Text(
            "DAQIRI receiver + GPU processing + DigitalMicrograph",
            color=MUTED,
            font_size=22,
        )
        group = (
            VGroup(eyebrow, title, subtitle)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.28)
            .to_edge(LEFT, buff=0.9)
        )
        accent = Rectangle(
            width=0.12, height=3.0, fill_color=ORANGE, fill_opacity=1, stroke_width=0
        ).next_to(group, LEFT, buff=0.28)
        badge = self.chip("128", "frames per natural bucket", ORANGE).to_edge(
            RIGHT, buff=1.0
        )
        self.play(
            FadeIn(accent, shift=UP * 0.2),
            FadeIn(group, shift=UP * 0.2),
            FadeIn(badge, shift=LEFT * 0.3),
            run_time=1.1,
        )
        self.wait(2.2)
        self.clear_scene()

    def acquisition_segment(self) -> None:
        title = self.title("Receive, place, assemble", "01 / acquisition")
        self.play(FadeIn(title))

        lane_group = VGroup()
        dots = VGroup()
        lane_paths = []
        for index in range(8):
            y = 2.35 - index * 0.52
            source = Text("RX{}".format(index), color=MUTED, font_size=14).move_to(
                LEFT * 6.35 + UP * y
            )
            path = Line(
                LEFT * 5.75 + UP * y,
                LEFT * 2.15 + UP * y,
                color="#35505d",
                stroke_width=2,
            )
            dot = Dot(
                path.get_start(), radius=0.055, color=CYAN if index < 4 else ORANGE
            )
            lane_group.add(source, path)
            dots.add(dot)
            lane_paths.append(path)

        igx = RoundedRectangle(
            width=2.25,
            height=4.9,
            corner_radius=0.18,
            fill_color=PANEL_ALT,
            fill_opacity=1,
            stroke_color=CYAN,
            stroke_width=2,
        ).move_to(LEFT * 0.8 + UP * 0.55)
        igx_label = (
            VGroup(
                Text("IGX", color=CREAM, font_size=31, weight="BOLD"),
                Text("DAQIRI RX", color=CYAN, font_size=16),
                Text("GPU memory", color=MUTED, font_size=14),
            )
            .arrange(DOWN, buff=0.12)
            .move_to(igx)
        )

        detector = RoundedRectangle(
            width=3.6,
            height=2.7,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=ORANGE,
            stroke_width=2,
        ).move_to(RIGHT * 4.35 + UP * 1.15)
        detector_label = Text(
            "tiled detector frame", color=CREAM, font_size=18
        ).next_to(detector, UP, buff=0.15)
        cells = VGroup()
        for row in range(4):
            for column in range(12):
                cell = Rectangle(
                    width=0.255,
                    height=0.49,
                    stroke_color="#35505d",
                    stroke_width=0.6,
                    fill_color=CYAN if column < 3 else ORANGE,
                    fill_opacity=0.76,
                )
                cell.move_to(
                    detector.get_corner(UP + LEFT)
                    + RIGHT * (0.29 + column * 0.27)
                    + DOWN * (0.38 + row * 0.53)
                )
                cells.add(cell)

        gpu_arrow = Arrow(
            igx.get_right(),
            detector.get_left(),
            buff=0.15,
            color=ORANGE,
            stroke_width=4,
        )
        gpu_label = Text("header placement", color=MUTED, font_size=13).next_to(
            gpu_arrow, UP, buff=0.08
        )
        self.play(
            Create(lane_group),
            FadeIn(dots),
            FadeIn(igx),
            FadeIn(igx_label),
            GrowArrow(gpu_arrow),
            FadeIn(gpu_label),
            run_time=1.0,
        )
        self.play(
            LaggedStart(
                *[MoveAlongPath(dot, path) for dot, path in zip(dots, lane_paths)],
                lag_ratio=0.07,
            ),
            run_time=1.3,
        )
        self.play(
            FadeIn(detector),
            FadeIn(detector_label),
            LaggedStart(*[FadeIn(cell, scale=0.4) for cell in cells], lag_ratio=0.015),
            run_time=1.5,
        )

        stack = VGroup()
        for index in range(6):
            card = RoundedRectangle(
                width=2.4,
                height=0.58,
                corner_radius=0.08,
                fill_color=PANEL_ALT,
                fill_opacity=1,
                stroke_color=CYAN,
                stroke_width=1,
            )
            card.shift(RIGHT * 4.95 + DOWN * (0.62 + index * 0.18))
            stack.add(card)
        stack_label = Text("128-frame bucket", color=CREAM, font_size=18).next_to(
            stack, DOWN, buff=0.2
        )
        self.play(FadeIn(stack, shift=DOWN * 0.25), FadeIn(stack_label), run_time=0.8)

        performance = self.metadata.get("performance", {})
        chips = (
            VGroup(
                self.chip(
                    "{:.2f} Gb/s".format(
                        float(performance.get("measured_input_gbps", 0))
                    ),
                    "example measured input",
                    CYAN,
                ),
                self.chip(
                    "{:.2f} Mpps".format(
                        float(performance.get("measured_packet_rate_mpps", 0))
                    ),
                    "example packet rate",
                    ORANGE,
                ),
                self.chip(
                    "{:,.0f} fps".format(
                        float(performance.get("measured_assembled_fps", 0))
                    ),
                    "assembled frames",
                    BLUE,
                ),
            )
            .arrange(RIGHT, buff=0.18)
            .scale(0.82)
            .to_edge(DOWN, buff=0.88)
        )
        caption = self.caption(
            "UDP tiles land in GPU memory; 128 frames form one processing bucket."
        )
        self.play(FadeIn(chips, shift=UP * 0.15), FadeIn(caption), run_time=0.7)
        self.wait(2.0)
        self.clear_scene()

    def acquisition_strategy_segment(self) -> None:
        title = self.title(
            "Two paths across the EELS dynamic range",
            "02 / acquisition strategy",
        )
        left_panel = RoundedRectangle(
            width=5.9,
            height=4.55,
            corner_radius=0.16,
            fill_color=PANEL,
            fill_opacity=0.72,
            stroke_color="#526d78",
            stroke_width=1.4,
        ).move_to(LEFT * 3.25 + UP * 0.08)
        right_panel = RoundedRectangle(
            width=5.9,
            height=4.55,
            corner_radius=0.16,
            fill_color=PANEL,
            fill_opacity=0.72,
            stroke_color=CYAN,
            stroke_width=1.8,
        ).move_to(RIGHT * 3.25 + UP * 0.08)

        dual_header = (
            VGroup(
                Text("DUAL EELS", color=CREAM, font_size=18, weight="BOLD"),
                Text("two optimized exposures", color=MUTED, font_size=13),
            )
            .arrange(DOWN, buff=0.05)
            .move_to(left_panel.get_top() + DOWN * 0.46)
        )
        tiled_header = (
            VGroup(
                Text(
                    "TILED STREAMING READOUT", color=CREAM, font_size=18, weight="BOLD"
                ),
                Text("one continuous region-aware stream", color=CYAN, font_size=13),
            )
            .arrange(DOWN, buff=0.05)
            .move_to(right_panel.get_top() + DOWN * 0.46)
        )

        low_loss = RoundedRectangle(
            width=3.15,
            height=0.72,
            corner_radius=0.1,
            fill_color=PANEL_ALT,
            fill_opacity=1,
            stroke_color=ORANGE,
            stroke_width=1.6,
        ).move_to(LEFT * 3.25 + UP * 0.8)
        low_loss_text = (
            VGroup(
                Text("LOW-LOSS EXPOSURE", color=ORANGE, font_size=15, weight="BOLD"),
                Text("short exposure - ZLP", color=MUTED, font_size=12),
            )
            .arrange(DOWN, buff=0.03)
            .move_to(low_loss)
        )
        core_loss = RoundedRectangle(
            width=3.15,
            height=0.72,
            corner_radius=0.1,
            fill_color=PANEL_ALT,
            fill_opacity=1,
            stroke_color=BLUE,
            stroke_width=1.6,
        ).move_to(LEFT * 3.25 + DOWN * 0.4)
        core_loss_text = (
            VGroup(
                Text("CORELOSS EXPOSURE", color=BLUE, font_size=15, weight="BOLD"),
                Text("long exposure - weak edges", color=MUTED, font_size=12),
            )
            .arrange(DOWN, buff=0.03)
            .move_to(core_loss)
        )
        dual_switch = Arrow(
            low_loss.get_bottom(),
            core_loss.get_top(),
            buff=0.08,
            color=MUTED,
            stroke_width=2.5,
        )
        switch_label = Text("switch energy range", color=MUTED, font_size=12).next_to(
            dual_switch, RIGHT, buff=0.12
        )
        splice = RoundedRectangle(
            width=2.45,
            height=0.58,
            corner_radius=0.09,
            fill_color="#203743",
            fill_opacity=1,
            stroke_color=CREAM,
            stroke_width=1.2,
        ).move_to(LEFT * 3.25 + DOWN * 1.48)
        splice_text = Text(
            "ALIGN + SPLICE", color=CREAM, font_size=14, weight="BOLD"
        ).move_to(splice)
        splice_arrow = Arrow(
            core_loss.get_bottom(),
            splice.get_top(),
            buff=0.08,
            color=MUTED,
            stroke_width=2.5,
        )

        detector = RoundedRectangle(
            width=4.85,
            height=2.15,
            corner_radius=0.1,
            fill_color="#0b1b23",
            fill_opacity=1,
            stroke_color="#526d78",
            stroke_width=1.2,
        ).move_to(RIGHT * 3.25 + UP * 0.12)
        detector_left = detector.get_left()[0]
        detector_top = detector.get_top()[1]
        zlp_tiles = VGroup()
        for index in range(4):
            tile = Rectangle(
                width=0.34,
                height=1.5,
                fill_color=ORANGE,
                fill_opacity=0.65 + index * 0.07,
                stroke_color=ORANGE,
                stroke_width=0.8,
            ).move_to(
                RIGHT * (detector_left + 0.36 + index * 0.38)
                + UP * (detector_top - 1.17)
            )
            zlp_tiles.add(tile)
        core_tiles = VGroup()
        for row in range(3):
            for column in range(6):
                tile = Rectangle(
                    width=0.43,
                    height=0.45,
                    fill_color=CYAN,
                    fill_opacity=0.52 + 0.04 * ((row + column) % 3),
                    stroke_color="#58a99f",
                    stroke_width=0.55,
                ).move_to(
                    RIGHT * (detector_left + 2.05 + column * 0.46)
                    + UP * (detector_top - 0.65 - row * 0.49)
                )
                core_tiles.add(tile)
        zlp_label = Text(
            "4 × ZLP READS", color=ORANGE, font_size=14, weight="BOLD"
        ).move_to(RIGHT * (detector_left + 0.94) + UP * (detector_top - 0.2))
        core_label = Text(
            "CoreLoss tiles", color=CYAN, font_size=15, weight="BOLD"
        ).move_to(RIGHT * (detector_left + 3.18) + UP * (detector_top - 0.2))
        gpu = RoundedRectangle(
            width=2.5,
            height=0.58,
            corner_radius=0.09,
            fill_color=PANEL_ALT,
            fill_opacity=1,
            stroke_color=CYAN,
            stroke_width=1.4,
        ).move_to(RIGHT * 3.25 + DOWN * 1.48)
        gpu_text = Text(
            "GPU ASSEMBLY", color=CREAM, font_size=14, weight="BOLD"
        ).move_to(gpu)
        gpu_arrow = Arrow(
            detector.get_bottom(),
            gpu.get_top(),
            buff=0.08,
            color=CYAN,
            stroke_width=2.8,
        )

        self.play(
            FadeIn(title),
            FadeIn(left_panel),
            FadeIn(right_panel),
            FadeIn(dual_header),
            FadeIn(tiled_header),
            run_time=0.8,
        )
        self.play(FadeIn(low_loss), FadeIn(low_loss_text), run_time=0.55)
        self.play(GrowArrow(dual_switch), FadeIn(switch_label), run_time=0.45)
        self.play(FadeIn(core_loss), FadeIn(core_loss_text), run_time=0.55)
        self.play(
            GrowArrow(splice_arrow), FadeIn(splice), FadeIn(splice_text), run_time=0.55
        )
        self.play(
            FadeIn(detector), FadeIn(zlp_label), FadeIn(core_label), run_time=0.45
        )
        self.play(
            LaggedStart(*[FadeIn(tile) for tile in zlp_tiles], lag_ratio=0.16),
            LaggedStart(*[FadeIn(tile) for tile in core_tiles], lag_ratio=0.025),
            run_time=1.0,
        )
        self.play(GrowArrow(gpu_arrow), FadeIn(gpu), FadeIn(gpu_text), run_time=0.55)
        caveat = Text(
            "Architectural comparison | numerical speedup not yet measured",
            color=MUTED,
            font_size=13,
        ).to_edge(DOWN, buff=0.91)
        caption = self.caption(
            "Different readout rates; full detector rows remain available before reduction."
        )
        self.play(FadeIn(caveat), FadeIn(caption), run_time=0.55)
        self.wait(2.2)
        self.clear_scene()

    def processing_segment(self) -> None:
        title = self.title("One ordered GPU correction path", "03 / processing")
        stages = [
            ("raw_frame.png", "RAW", CYAN),
            ("dark_subtracted_frame.png", "DARK SUBTRACTED", BLUE),
            ("blr_corrected_frame.png", "BLR CORRECTED", ORANGE),
            ("mask_overlay.png", "MASK DECISION", RED),
            ("corrected_frame.png", "FULLY CORRECTED", CYAN),
        ]
        rail_items = VGroup()
        for _filename, label, color in stages:
            text = Text(label, color=color, font_size=14, weight="BOLD")
            rail_items.add(text)
        rail_items.arrange(RIGHT, buff=0.3).to_edge(UP, buff=1.32)
        connectors = VGroup(
            *[
                Arrow(
                    rail_items[i].get_right(),
                    rail_items[i + 1].get_left(),
                    buff=0.08,
                    color="#526d78",
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.18,
                )
                for i in range(len(rail_items) - 1)
            ]
        )
        self.play(FadeIn(title), FadeIn(rail_items), Create(connectors), run_time=0.8)

        card = None
        marker = None
        for index, (filename, label, color) in enumerate(stages):
            next_card = self.image_card(filename, width=10.8).shift(DOWN * 0.28)
            next_marker = RoundedRectangle(
                width=rail_items[index].width + 0.22,
                height=rail_items[index].height + 0.16,
                corner_radius=0.08,
                stroke_color=color,
                stroke_width=2,
                fill_opacity=0,
            ).move_to(rail_items[index])
            if card is None:
                self.play(
                    FadeIn(next_card, shift=UP * 0.18),
                    FadeIn(next_marker),
                    run_time=0.8,
                )
            else:
                self.play(
                    FadeOut(card),
                    FadeOut(marker),
                    FadeIn(next_card),
                    FadeIn(next_marker),
                    run_time=0.65,
                )
            card = next_card
            marker = next_marker
            self.wait(1.4 if label != "MASK DECISION" else 1.8)

        mask = self.metadata.get("mask", {})
        note = (
            self.chip(
                "{:.3f}%".format(100.0 * float(mask.get("fraction", 0))),
                "pixels masked in this bucket",
                ORANGE,
            )
            .scale(0.82)
            .to_corner(DOWN + RIGHT, buff=0.48)
        )
        caption = self.caption(
            "The offline assets use the same correction order and parameters as the runtime."
        )
        self.play(FadeIn(note), FadeIn(caption), run_time=0.6)
        self.wait(2.0)
        self.clear_scene()

    def spectrum_segment(self) -> None:
        title = self.title(
            "Reduce the bucket into an EELS spectrum", "04 / science product"
        )
        sum_card = self.image_card("bucket_sum.png", width=9.8).shift(UP * 0.15)
        bucket = VGroup(
            *[
                RoundedRectangle(
                    width=1.5,
                    height=0.42,
                    corner_radius=0.05,
                    fill_color=PANEL_ALT,
                    fill_opacity=1,
                    stroke_color=CYAN,
                    stroke_width=0.8,
                ).shift(LEFT * 5.85 + UP * (1.45 - i * 0.13))
                for i in range(7)
            ]
        )
        bucket_text = (
            Text("128 corrected\nframes", color=CREAM, font_size=19, line_spacing=0.8)
            .scale_to_fit_width(1.35)
            .next_to(bucket, DOWN, buff=0.2)
        )
        arrow = Arrow(
            bucket.get_right(),
            sum_card.get_left(),
            buff=0.2,
            color=ORANGE,
            stroke_width=4,
        )
        self.play(
            FadeIn(title),
            FadeIn(bucket),
            FadeIn(bucket_text),
            GrowArrow(arrow),
            FadeIn(sum_card),
            run_time=1.1,
        )
        self.wait(1.6)

        linear = self.image_card("spectrum_linear.png", width=11.4).shift(DOWN * 0.15)
        self.play(
            FadeOut(bucket),
            FadeOut(bucket_text),
            FadeOut(arrow),
            FadeOut(sum_card),
            FadeIn(linear, shift=UP * 0.15),
            run_time=0.9,
        )
        caption = self.caption(
            "Rows integrate; four repeated zero-loss regions fold into one physical spectrum."
        )
        self.play(FadeIn(caption))
        self.wait(2.2)
        log_plot = self.image_card("spectrum_log.png", width=11.4).shift(DOWN * 0.15)
        self.play(FadeOut(linear), FadeIn(log_plot), run_time=0.8)
        log_label = (
            self.chip("LOG Y", "reveals the full dynamic range", ORANGE)
            .scale(0.8)
            .to_corner(UP + RIGHT, buff=0.55)
            .shift(DOWN * 0.55)
        )
        self.play(
            FadeIn(log_label),
            Indicate(log_plot, color=ORANGE, scale_factor=1.01),
            run_time=0.7,
        )
        self.wait(2.6)
        self.clear_scene()

    def outputs_segment(self) -> None:
        title = self.title("Full bursts and a thin live view", "05 / outputs + control")
        gpu = RoundedRectangle(
            width=2.6,
            height=2.0,
            corner_radius=0.16,
            fill_color=PANEL_ALT,
            fill_opacity=1,
            stroke_color=CYAN,
            stroke_width=2,
        ).move_to(LEFT * 4.7 + UP * 0.6)
        gpu_text = (
            VGroup(
                Text("GPU PIPELINE", color=CREAM, font_size=21, weight="BOLD"),
                Text("128-frame bucket", color=MUTED, font_size=14),
            )
            .arrange(DOWN, buff=0.12)
            .move_to(gpu)
        )

        disk = VGroup(
            *[
                RoundedRectangle(
                    width=2.8,
                    height=0.48,
                    corner_radius=0.07,
                    fill_color=PANEL,
                    fill_opacity=1,
                    stroke_color=ORANGE,
                    stroke_width=1,
                ).shift(RIGHT * 0.1 + UP * (1.78 - i * 0.15))
                for i in range(5)
            ]
        )
        disk_label = (
            VGroup(
                Text("HDF5 BURST", color=ORANGE, font_size=20, weight="BOLD"),
                Text("selected full buckets", color=MUTED, font_size=14),
            )
            .arrange(DOWN, buff=0.1)
            .next_to(disk, DOWN, buff=0.18)
        )

        dm = RoundedRectangle(
            width=4.15,
            height=3.4,
            corner_radius=0.15,
            fill_color="#d9dde0",
            fill_opacity=1,
            stroke_color=CREAM,
            stroke_width=2,
        ).move_to(RIGHT * 4.55 + UP * 0.45)
        dm_bar = (
            Rectangle(
                width=4.15,
                height=0.42,
                fill_color="#44535b",
                fill_opacity=1,
                stroke_width=0,
            )
            .move_to(dm)
            .align_to(dm, UP)
        )
        dm_title = Text("DigitalMicrograph", color=CREAM, font_size=14).move_to(dm_bar)
        dm_image = (
            ImageMobject(str(self.assets / "corrected_frame.png"))
            .set_width(3.65)
            .move_to(dm)
            .shift(UP * 0.18)
        )
        dm_controls = (
            VGroup(
                *[
                    RoundedRectangle(
                        width=0.8,
                        height=0.26,
                        corner_radius=0.04,
                        fill_color=CYAN if i == 0 else PANEL_ALT,
                        fill_opacity=1,
                        stroke_width=0,
                    )
                    for i in range(4)
                ]
            )
            .arrange(RIGHT, buff=0.08)
            .next_to(dm_image, DOWN, buff=0.12)
        )
        dm_group = Group(dm, dm_bar, dm_title, dm_image, dm_controls)

        burst_arrow = Arrow(
            gpu.get_right(), disk.get_left(), buff=0.18, color=ORANGE, stroke_width=4
        )
        thin_arrow = Arrow(
            gpu.get_right(), dm.get_left(), buff=0.18, color=CYAN, stroke_width=4
        )
        burst_text = Text("burst writer", color=ORANGE, font_size=13).next_to(
            burst_arrow, UP, buff=0.08
        )
        thin_text = Text("ZeroMQ PUB: frame + sum", color=CYAN, font_size=13).next_to(
            thin_arrow, DOWN, buff=0.08
        )
        control_arrow = Arrow(
            dm.get_left() + DOWN * 1.25,
            gpu.get_right() + DOWN * 0.78,
            buff=0.15,
            color=BLUE,
            stroke_width=3,
            path_arc=-0.4,
        )
        control_text = Text(
            "ZeroMQ REQ/REP controls", color=BLUE, font_size=13
        ).next_to(control_arrow, DOWN, buff=0.05)

        self.play(FadeIn(title), FadeIn(gpu), FadeIn(gpu_text), run_time=0.7)
        self.play(
            GrowArrow(burst_arrow),
            FadeIn(burst_text),
            FadeIn(disk),
            FadeIn(disk_label),
            run_time=0.9,
        )
        self.play(
            GrowArrow(thin_arrow), FadeIn(thin_text), FadeIn(dm_group), run_time=1.0
        )
        self.play(GrowArrow(control_arrow), FadeIn(control_text), run_time=0.9)
        caption = self.caption(
            "The live display is thinned; requested full buckets are retained without streaming every frame."
        )
        self.play(FadeIn(caption))
        self.wait(3.0)
        self.clear_scene()

    def outro_segment(self) -> None:
        title = Text(
            "From detector packets to live EELS products",
            color=CREAM,
            font_size=40,
            weight="BOLD",
        )
        flow = Text(
            "100 GbE  ->  GPU assembly  ->  corrections  ->  EELS  ->  DM",
            color=CYAN,
            font_size=23,
        )
        note = Text(
            "Recorded NiO example | reproducible offline assets | live-ready architecture",
            color=MUTED,
            font_size=17,
        )
        group = VGroup(title, flow, note).arrange(DOWN, buff=0.34)
        self.play(FadeIn(group, shift=UP * 0.2), run_time=1.0)
        self.wait(2.8)
        self.play(FadeOut(group), run_time=0.7)


class AcquisitionScene(DemoScene):
    def construct(self):
        self.acquisition_segment()


class AcquisitionStrategyScene(DemoScene):
    def construct(self):
        self.acquisition_strategy_segment()


class ProcessingScene(DemoScene):
    def construct(self):
        self.processing_segment()


class SpectrumScene(DemoScene):
    def construct(self):
        self.spectrum_segment()


class OutputsScene(DemoScene):
    def construct(self):
        self.outputs_segment()


class FullDemo(DemoScene):
    def construct(self):
        self.intro_segment()
        self.acquisition_segment()
        self.acquisition_strategy_segment()
        self.processing_segment()
        self.spectrum_segment()
        self.outputs_segment()
        self.outro_segment()
