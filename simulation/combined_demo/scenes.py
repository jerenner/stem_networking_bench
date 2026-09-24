"""Manim scenes combining the LMTO DOEELS workflow with DAQIRI acquisition."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Annulus,
    ArcBetweenPoints,
    Arrow,
    Circle,
    Create,
    CubicBezier,
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
    linear,
)
from PIL import Image

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
TEXT_FONT = os.environ.get("LMTO_COMBINED_FONT", "Avenir Next")

config.background_color = BACKGROUND
Text.set_default(font=TEXT_FONT, disable_ligatures=False)


def _asset_directory() -> Path:
    configured = os.environ.get("LMTO_COMBINED_ASSETS")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "demo" / "assets" / "generated_200keV"


def phase_tile_rows(source: int, phase: int) -> tuple[int, int]:
    """Return the ZLP and CoreLoss tile rows for one sketched readout phase."""
    if not 0 <= source < 8 or not 0 <= phase < 4:
        raise ValueError("source must be 0..7 and phase must be 0..3")
    channel = source % 4
    zlp_row = channel if source < 4 else 7 - channel
    core_row = phase * 4 + channel if source < 4 else 31 - phase * 4 - channel
    return zlp_row, core_row


class CombinedDemoScene(Scene):
    """Shared visual system and segments for the combined movie."""

    def setup(self) -> None:
        super().setup()
        self.assets = _asset_directory()
        metadata_path = self.assets / "demo_metadata.json"
        required = (
            "final_haadf.png",
            "final_mn.png",
            "final_o.png",
            "final_ti.png",
            "probe_raster.png",
            "raw_detector_frame.png",
            "analog_vs_counted_processing.png",
        )
        missing = [name for name in required if not (self.assets / name).exists()]
        if not metadata_path.exists() or missing:
            detail = ", ".join(missing) if missing else metadata_path.name
            raise FileNotFoundError(
                f"Combined-demo assets are missing ({detail}). "
                "Run combined_demo/render_demo.sh assets first."
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

    def stage_box(
        self,
        label: str,
        detail: str,
        color: str,
        width: float = 2.35,
        height: float = 1.12,
    ) -> VGroup:
        heading = Text(label, color=color, font_size=17, weight="BOLD")
        note = Text(detail, color=MUTED, font_size=11)
        contents = VGroup(heading, note).arrange(DOWN, buff=0.08)
        box = RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=1.0,
            stroke_color=color,
            stroke_width=1.4,
        )
        contents.move_to(box)
        return VGroup(box, contents)

    def chip(self, value: str, label: str, color: str, width: float = 2.45) -> VGroup:
        number = Text(value, color=color, font_size=24, weight="BOLD")
        description = Text(label, color=MUTED, font_size=12)
        contents = VGroup(number, description).arrange(DOWN, buff=0.04)
        box = RoundedRectangle(
            width=width,
            height=0.84,
            corner_radius=0.1,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color="#35505d",
            stroke_width=1,
        )
        contents.move_to(box)
        return VGroup(box, contents)

    def image_card(self, filename: str, width: float, shift=ORIGIN) -> Group:
        image = ImageMobject(str(self.assets / filename)).set_width(width).shift(shift)
        frame = RoundedRectangle(
            width=image.width + 0.1,
            height=image.height + 0.1,
            corner_radius=0.07,
            fill_opacity=0,
            stroke_color="#35505d",
            stroke_width=1.0,
        ).move_to(image)
        return Group(frame, image)

    def clear_scene(self, run_time: float = 0.35) -> None:
        if self.mobjects:
            self.play(FadeOut(Group(*self.mobjects)), run_time=run_time)

    def native_tiled_frame(self, width: float = 6.6) -> tuple[Group, list[list[VGroup]]]:
        """Overlay the intended two-FPGA tile schedule on a raw frame."""
        with Image.open(self.assets / "raw_detector_frame.png") as source:
            pixels = np.asarray(source.convert("RGBA"))
        height_px, width_px = pixels.shape[:2]
        detector = pixels[
            int(0.045 * height_px) : int(0.415 * height_px),
            int(0.042 * width_px) : int(0.961 * width_px),
        ]
        image = ImageMobject(detector).set_width(width)
        image.stretch_to_fit_height(2.4)
        frame = Rectangle(
            width=width + 0.08,
            height=image.height + 0.08,
            fill_opacity=0,
            stroke_color=ORANGE,
            stroke_width=1.5,
        ).move_to(image)
        phase_source_covers = [[VGroup() for _ in range(8)] for _ in range(4)]
        grid = VGroup()

        def tile_rectangle(
            row: int,
            column: int,
            tile_rows: int,
            tile_columns: int,
            source: int,
            phase: int,
        ) -> Rectangle:
            tile_width = image.width / tile_columns
            tile_height = image.height / tile_rows
            cover = Rectangle(
                width=tile_width + 0.006,
                height=tile_height + 0.006,
                fill_color=BACKGROUND,
                fill_opacity=0.97,
                stroke_color=CYAN if source < 4 else ORANGE,
                stroke_width=0.18,
            )
            cover.move_to(
                image.get_corner(UP + LEFT)
                + RIGHT * (column + 0.5) * tile_width
                + DOWN * (row + 0.5) * tile_height
            )
            phase_source_covers[phase][source].add(cover)
            return cover

        covers = VGroup()
        # Match the collaborator's sketch: a complete 192-column ZLP read
        # appears alongside contiguous CoreLoss bands at the top and bottom.
        # Each phase adds four CoreLoss tile rows from each edge (8/32 = 25%)
        # and the next six ZLP tile columns (6/24 = one complete read).
        for source in range(8):
            for phase in range(4):
                zlp_row, core_row = phase_tile_rows(source, phase)
                for column in range(24):
                    covers.add(tile_rectangle(core_row, column + 6, 32, 30, source, phase))
                for column in range(6):
                    covers.add(tile_rectangle(zlp_row, phase * 6 + column, 8, 120, source, phase))

        left = image.get_left()[0]
        right = image.get_right()[0]
        top = image.get_top()[1]
        bottom = image.get_bottom()[1]
        zlp_right = left + image.width * 0.2
        for column in range(25):
            x = left + image.width * 0.2 * column / 24
            grid.add(Line([x, bottom, 0], [x, top, 0], color="#55707a", stroke_width=0.35))
        for row in range(9):
            y = top - image.height * row / 8
            grid.add(Line([left, y, 0], [zlp_right, y, 0], color="#55707a", stroke_width=0.4))
        for column in range(25):
            x = zlp_right + image.width * 0.8 * column / 24
            grid.add(Line([x, bottom, 0], [x, top, 0], color="#55707a", stroke_width=0.35))
        for row in range(33):
            y = top - image.height * row / 32
            grid.add(Line([zlp_right, y, 0], [right, y, 0], color="#55707a", stroke_width=0.35))
        for lane in range(5):
            x = left + image.width * 0.05 * lane
            grid.add(Line([x, bottom, 0], [x, top, 0], color=ORANGE, stroke_width=1.0))
        return Group(image, covers, grid, frame), phase_source_covers

    def map_pixel_covers(
        self, image: ImageMobject, rows: int = 16, columns: int = 16
    ) -> list[VGroup]:
        """Cover a map with one rectangle per simulated raster position."""
        plot_width = image.width * 0.934
        plot_height = image.height * 0.934
        plot_center = image.get_center() + DOWN * image.height * 0.025
        covers = []
        for row in range(rows):
            cover_row = VGroup()
            for column in range(columns):
                cover = Rectangle(
                    width=plot_width / columns + 0.008,
                    height=plot_height / rows + 0.008,
                    fill_color=BACKGROUND,
                    fill_opacity=1,
                    stroke_width=0,
                )
                cover.move_to(
                    plot_center
                    + LEFT * plot_width / 2
                    + RIGHT * (column + 0.5) * plot_width / columns
                    + UP * plot_height / 2
                    + DOWN * (row + 0.5) * plot_height / rows
                )
                cover_row.add(cover)
            covers.append(cover_row)
        return covers

    def intro_segment(self) -> None:
        eyebrow = Text("DOEELS + DAQIRI", color=CYAN, font_size=18, weight="BOLD")
        heading = Text(
            "From electrons to live\nelemental maps",
            color=CREAM,
            font_size=49,
            weight="BOLD",
            line_spacing=0.88,
        )
        subtitle = Text(
            "Structure, chemistry, and acquisition in one workflow",
            color=MUTED,
            font_size=19,
        )
        copy = (
            VGroup(eyebrow, heading, subtitle)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            .to_edge(LEFT, buff=0.72)
        )
        accent = Rectangle(
            width=0.1,
            height=2.9,
            fill_color=ORANGE,
            fill_opacity=1,
            stroke_width=0,
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
        self.wait(2.8)
        self.clear_scene()

    def microscope_segment(self) -> None:
        title = self.title("One probe position produces structure and chemistry", "01 / microscope")
        # The physical column follows the collaborator's setup sketch: probe,
        # specimen, annular HAADF, entrance aperture, prism, then silicon.
        column_x = -3.55
        source = Circle(
            radius=0.28,
            fill_color=PANEL_ALT,
            fill_opacity=1,
            stroke_color=CYAN,
            stroke_width=2,
        ).move_to([column_x, 2.3, 0])
        source_label = Text("e-", color=CYAN, font_size=18, weight="BOLD").move_to(source)
        beam_axis = Line(
            [column_x, 2.02, 0], [column_x, -0.58, 0], color="#35505d", stroke_width=1.2
        )
        convergence = VGroup(
            Line([column_x - 0.27, 2.01, 0], [column_x - 0.06, 1.6, 0], color=CYAN),
            Line([column_x + 0.27, 2.01, 0], [column_x + 0.06, 1.6, 0], color=CYAN),
        )
        scan_coils = VGroup(
            Rectangle(
                width=0.42,
                height=0.16,
                fill_color=BLUE,
                fill_opacity=0.85,
                stroke_width=0,
            ).move_to([column_x - 0.36, 1.83, 0]),
            Rectangle(
                width=0.42,
                height=0.16,
                fill_color=BLUE,
                fill_opacity=0.85,
                stroke_width=0,
            ).move_to([column_x + 0.36, 1.83, 0]),
        )
        probe_label = Text("focused probe", color=MUTED, font_size=12).move_to(
            [column_x + 1.35, 2.08, 0]
        )
        sample_box = RoundedRectangle(
            width=3.3,
            height=0.72,
            corner_radius=0.12,
            fill_color=PANEL,
            fill_opacity=0.7,
            stroke_color=YELLOW,
            stroke_width=1.5,
        ).move_to([column_x, 1.24, 0])
        atoms = VGroup()
        atom_colors = (YELLOW, GREEN, BLUE, RED, BLUE)
        for row in range(2):
            for column in range(7):
                atom = Dot(
                    radius=0.09 + 0.015 * ((row + column) % 2),
                    color=atom_colors[(row * 7 + column) % len(atom_colors)],
                )
                atom.move_to(
                    sample_box.get_corner(UP + LEFT)
                    + RIGHT * (0.39 + column * 0.42)
                    + DOWN * (0.22 + row * 0.27)
                )
                atoms.add(atom)
        sample_label = Text("LMTO lattice", color=CREAM, font_size=13).next_to(
            sample_box, LEFT, buff=0.18
        )
        haadf = Annulus(
            inner_radius=0.2,
            outer_radius=0.48,
            fill_color=CREAM,
            fill_opacity=0.75,
            stroke_color=CREAM,
            stroke_width=1.0,
        ).move_to([column_x, 0.17, 0])
        haadf_label = Text("annular HAADF", color=CREAM, font_size=13).next_to(
            haadf, LEFT, buff=0.18
        )
        haadf_signal = self.stage_box("STRUCTURE", "HAADF intensity", CREAM, 2.35, 0.86).move_to(
            [-0.48, 0.17, 0]
        )
        haadf_readout = Arrow(
            haadf.get_right(), haadf_signal.get_left(), buff=0.1, color=CREAM, stroke_width=2.4
        )
        entrance = Circle(
            radius=0.13,
            fill_color=BACKGROUND,
            fill_opacity=1,
            stroke_color=ORANGE,
            stroke_width=2,
        ).move_to([column_x, -0.55, 0])
        entrance_label = Text("EELS entrance aperture", color=MUTED, font_size=11).next_to(
            entrance, LEFT, buff=0.16
        )
        spectrometer = self.stage_box(
            "MAGNETIC PRISM", "disperse by energy", ORANGE, 3.1, 0.76
        ).move_to([column_x, -1.1, 0])
        silicon_frame = RoundedRectangle(
            width=3.45,
            height=0.74,
            corner_radius=0.08,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=BLUE,
            stroke_width=2,
        ).move_to([-2.12, -2.45, 0])
        silicon_pixels = VGroup()
        for row in range(3):
            for column in range(12):
                cell = Rectangle(
                    width=0.25,
                    height=0.15,
                    fill_color=BLUE if column < 2 else ORANGE,
                    fill_opacity=0.5 + 0.08 * ((row + column) % 3),
                    stroke_color="#35505d",
                    stroke_width=0.35,
                )
                cell.move_to(
                    silicon_frame.get_corner(UP + LEFT)
                    + RIGHT * (0.25 + column * 0.27)
                    + DOWN * (0.22 + row * 0.17)
                )
                silicon_pixels.add(cell)
        silicon_label = Text("silicon camera", color=BLUE, font_size=12).next_to(
            silicon_frame, LEFT, buff=0.16
        )
        detector_region_labels = VGroup(
            Text("ZLP", color=BLUE, font_size=11, weight="BOLD").move_to(
                silicon_frame.get_top() + DOWN * 0.11 + LEFT * 1.35
            ),
            Text("CoreLoss", color=ORANGE, font_size=11, weight="BOLD").move_to(
                silicon_frame.get_top() + DOWN * 0.11 + RIGHT * 0.48
            ),
        )
        detector_divider = Line(
            silicon_frame.get_top() + LEFT * 1.035,
            silicon_frame.get_bottom() + LEFT * 1.035,
            color=CREAM,
            stroke_width=1.1,
        )
        prism_exit = spectrometer.get_bottom()
        zlp_landing = np.array([prism_exit[0], silicon_frame.get_top()[1], 0.0])
        dispersed_paths = VGroup(
            Line(
                prism_exit,
                zlp_landing,
                color=BLUE,
                stroke_width=3.0,
            ),
            CubicBezier(
                prism_exit,
                prism_exit + DOWN * 0.24,
                silicon_frame.get_top() + LEFT * 1.19 + UP * 0.12,
                silicon_frame.get_top() + LEFT * 1.1,
                color=CYAN,
                stroke_width=3.0,
            ),
            CubicBezier(
                prism_exit,
                prism_exit + DOWN * 0.24,
                silicon_frame.get_top() + LEFT * 0.08 + UP * 0.04,
                silicon_frame.get_top() + RIGHT * 0.72,
                color=ORANGE,
                stroke_width=3.0,
            ),
        )
        dispersion_note = Text(
            "more energy loss\nmore deflection",
            color=MUTED,
            font_size=12,
            line_spacing=0.9,
        ).move_to([0.05, -1.58, 0])
        daqiri = self.stage_box("DAQIRI", "packets to GPU", CYAN, 2.05, 0.92).move_to(
            [1.63, -2.45, 0]
        )
        live = self.stage_box("LIVE PRODUCTS", "spectra + maps", GREEN, 2.35, 0.92).move_to(
            [4.73, -2.45, 0]
        )
        camera_to_daqiri = Arrow(
            silicon_frame.get_right(), daqiri.get_left(), buff=0.08, color=CYAN, stroke_width=2.7
        )
        daqiri_to_live = Arrow(
            daqiri.get_right(), live.get_left(), buff=0.08, color=GREEN, stroke_width=2.7
        )
        incident_path = Line(
            source.get_bottom(), sample_box.get_top(), color=CYAN, stroke_width=2.5
        )
        transmitted_path = Line(
            sample_box.get_bottom(), entrance.get_center(), color=ORANGE, stroke_width=2.2
        )
        prism_input_path = Line(
            entrance.get_center(), spectrometer.get_top(), color=ORANGE, stroke_width=2.2
        )
        scattered_paths = VGroup(
            Line(sample_box.get_bottom(), haadf.get_center() + LEFT * 0.4, color=CREAM),
            Line(sample_box.get_bottom(), haadf.get_center() + RIGHT * 0.4, color=CREAM),
        )
        self.play(
            FadeIn(title),
            FadeIn(source),
            FadeIn(source_label),
            Create(beam_axis),
            Create(convergence),
            FadeIn(scan_coils),
            FadeIn(probe_label),
            FadeIn(sample_box),
            FadeIn(atoms),
            FadeIn(sample_label),
            run_time=0.9,
        )
        electrons = VGroup(
            *[Dot(incident_path.get_start(), radius=0.045, color=CYAN) for _ in range(7)]
        )
        self.add(electrons)
        self.play(
            LaggedStart(
                *[MoveAlongPath(electron, incident_path, run_time=0.8) for electron in electrons],
                lag_ratio=0.12,
            ),
            run_time=1.4,
        )
        self.play(
            FadeIn(haadf),
            FadeIn(haadf_label),
            Create(scattered_paths),
            FadeIn(entrance),
            FadeIn(entrance_label),
            Create(transmitted_path),
            Create(prism_input_path),
            FadeIn(spectrometer),
            Create(dispersed_paths),
            FadeIn(dispersion_note),
            FadeIn(silicon_frame),
            FadeIn(silicon_pixels),
            Create(detector_divider),
            FadeIn(detector_region_labels),
            FadeIn(silicon_label),
            FadeIn(daqiri),
            FadeIn(live),
            GrowArrow(haadf_readout),
            FadeIn(haadf_signal),
            GrowArrow(camera_to_daqiri),
            GrowArrow(daqiri_to_live),
            run_time=1.1,
        )
        scattered_electrons = VGroup(
            Dot(sample_box.get_bottom(), radius=0.05, color=CREAM),
            Dot(sample_box.get_bottom(), radius=0.05, color=CREAM),
        )
        transmitted_electrons = VGroup(
            *[Dot(sample_box.get_bottom(), radius=0.045, color=ORANGE) for _ in range(4)]
        )
        self.add(scattered_electrons, transmitted_electrons)
        self.play(
            MoveAlongPath(scattered_electrons[0], scattered_paths[0]),
            MoveAlongPath(scattered_electrons[1], scattered_paths[1]),
            LaggedStart(
                *[
                    MoveAlongPath(electron, transmitted_path, run_time=0.75)
                    for electron in transmitted_electrons
                ],
                lag_ratio=0.12,
            ),
            run_time=1.0,
        )
        self.play(
            LaggedStart(
                *[
                    MoveAlongPath(electron, prism_input_path, run_time=0.65)
                    for electron in transmitted_electrons
                ],
                lag_ratio=0.1,
            ),
            run_time=0.9,
        )
        dispersed_electrons = VGroup(
            *[
                Dot(path.get_start(), radius=0.045, color=color)
                for path, color in zip(dispersed_paths, (BLUE, CYAN, ORANGE))
                for _ in range(2)
            ]
        )
        self.add(dispersed_electrons)
        self.play(
            LaggedStart(
                *[
                    MoveAlongPath(electron, dispersed_paths[index // 2], run_time=0.75)
                    for index, electron in enumerate(dispersed_electrons)
                ],
                lag_ratio=0.1,
            ),
            run_time=1.2,
        )
        caption = self.caption(
            "HAADF catches high-angle electrons; transmitted electrons are energy-dispersed onto silicon."
        )
        self.play(FadeIn(caption))
        self.wait(2.6)
        self.clear_scene()

    def scan_segment(self) -> None:
        title = self.title("The same two measurements follow the probe raster", "02 / scan")
        raster = self.image_card("probe_raster.png", width=6.8, shift=LEFT * 2.9 + DOWN * 0.05)
        scan = self.metadata["scan"]
        chips = (
            VGroup(
                self.chip("16 x 16", "registered probe positions", CYAN, 3.2),
                self.chip(
                    f"{scan['integrations_per_position']:,} frames",
                    "single-frame readouts per point",
                    BLUE,
                    3.2,
                ),
                self.chip(f"{scan['dwell_ms']:.1f} ms", "dwell per point", ORANGE, 3.2),
                self.chip(
                    "> 6 million",
                    "incident electrons per point",
                    GREEN,
                    3.2,
                ),
            )
            .arrange(DOWN, buff=0.18)
            .move_to(RIGHT * 4.45 + DOWN * 0.05)
        )
        self.play(FadeIn(title), FadeIn(raster), FadeIn(chips), run_time=0.9)
        caption = self.caption(
            "Each (x, y) position keeps its HAADF intensity and complete EELS readout registered."
        )
        self.play(FadeIn(caption))
        self.wait(3.4)
        self.clear_scene()

    def packet_segment(self) -> None:
        title = self.title("Native detector tiles assemble directly in GPU memory", "03 / DAQIRI")
        source_labels = VGroup()
        lane_paths = VGroup()
        packet_dots = VGroup()
        fpga_colors = (CYAN, ORANGE)
        for index in range(8):
            y = 2.1 - index * 0.48
            group_color = fpga_colors[index // 4]
            label = Text(f"RX{index}", color=group_color, font_size=12, weight="BOLD").move_to(
                LEFT * 6.25 + UP * y
            )
            lane = Line(
                LEFT * 5.85 + UP * y,
                LEFT * 3.25 + UP * y,
                color="#35505d",
                stroke_width=2,
            )
            packet = RoundedRectangle(
                width=0.32,
                height=0.13,
                corner_radius=0.02,
                fill_color=group_color,
                fill_opacity=1,
                stroke_width=0,
            ).move_to(lane.get_start())
            source_labels.add(label)
            lane_paths.add(lane)
            packet_dots.add(packet)
        fpga_panels = VGroup(
            RoundedRectangle(
                width=3.68,
                height=1.92,
                corner_radius=0.1,
                fill_color=CYAN,
                fill_opacity=0.05,
                stroke_color=CYAN,
                stroke_width=1.0,
            ).move_to([-4.82, 1.38, 0]),
            RoundedRectangle(
                width=3.68,
                height=1.92,
                corner_radius=0.1,
                fill_color=ORANGE,
                fill_opacity=0.05,
                stroke_color=ORANGE,
                stroke_width=1.0,
            ).move_to([-4.82, -0.54, 0]),
        )
        fpga_labels = VGroup(
            Text("FPGA 0", color=CYAN, font_size=11, weight="BOLD").move_to([-5.7, 2.43, 0]),
            Text("FPGA 1", color=ORANGE, font_size=11, weight="BOLD").move_to([-5.7, 0.51, 0]),
        )
        gpu = RoundedRectangle(
            width=1.75,
            height=4.35,
            corner_radius=0.16,
            fill_color=PANEL_ALT,
            fill_opacity=1,
            stroke_color=CYAN,
            stroke_width=2,
        ).move_to(LEFT * 2.15 + UP * 0.42)
        gpu_label = (
            VGroup(
                Text("DAQIRI", color=CYAN, font_size=22, weight="BOLD"),
                Text("RX", color=CREAM, font_size=18, weight="BOLD"),
                Text("GPU memory", color=MUTED, font_size=12),
            )
            .arrange(DOWN, buff=0.1)
            .move_to(gpu)
        )
        tiled_frame, phase_source_covers = self.native_tiled_frame(width=6.6)
        tiled_frame.move_to(RIGHT * 3.25 + UP * 0.42)
        image = tiled_frame[0]
        zlp_label = Text(
            "4 ZLP lanes\n192 tiles: 128 x 32",
            color=BLUE,
            font_size=11,
            line_spacing=0.86,
        ).move_to(image.get_top() + UP * 0.34 + LEFT * image.width * 0.4)
        core_label = Text(
            "CoreLoss\n768 tiles: 32 x 128",
            color=ORANGE,
            font_size=11,
            line_spacing=0.86,
        ).move_to(image.get_top() + UP * 0.34 + RIGHT * image.width * 0.1)
        frame_note = Text(
            "FPGA 0 / RX0-3: top half     FPGA 1 / RX4-7: bottom half",
            color=MUTED,
            font_size=10.5,
        ).move_to([3.25, -1.27, 0])
        header = RoundedRectangle(
            width=4.7,
            height=0.72,
            corner_radius=0.1,
            fill_color=PANEL,
            fill_opacity=1,
            stroke_color=ORANGE,
            stroke_width=1.2,
        ).move_to(RIGHT * 2.15 + DOWN * 1.88)
        header_text = Text(
            "header: frame ID  |  source  |  tile ordinal  |  payload",
            color=CREAM,
            font_size=13,
        ).move_to(header)
        self.play(
            FadeIn(title),
            FadeIn(fpga_panels),
            FadeIn(fpga_labels),
            FadeIn(source_labels),
            Create(lane_paths),
            FadeIn(packet_dots),
            FadeIn(gpu),
            FadeIn(gpu_label),
            FadeIn(tiled_frame),
            FadeIn(zlp_label),
            FadeIn(core_label),
            FadeIn(frame_note),
            run_time=0.9,
        )
        self.play(
            LaggedStart(
                *[
                    MoveAlongPath(packet, lane, run_time=0.75)
                    for packet, lane in zip(packet_dots, lane_paths)
                ],
                lag_ratio=0.08,
            ),
            run_time=1.4,
        )
        self.play(FadeIn(header), FadeIn(header_text), run_time=0.45)
        phase_label = None
        for phase, source_groups in enumerate(phase_source_covers):
            next_label = Text(
                f"ZLP read {phase + 1} complete  +  CoreLoss {(phase + 1) * 25}%",
                color=CREAM,
                font_size=12,
            ).move_to([3.25, -1.02, 0])
            animations = [
                AnimationGroup(
                    *[FadeOut(covers, run_time=0.55) for covers in source_groups],
                    lag_ratio=0,
                ),
                FadeIn(next_label),
            ]
            if phase_label is not None:
                animations.append(FadeOut(phase_label))
            self.play(*animations, run_time=0.65)
            phase_label = next_label
        caption = self.caption(
            "Top and bottom arrive together; each ZLP read completes with one CoreLoss quarter.",
            ORANGE,
        )
        self.play(FadeIn(caption))
        self.wait(2.4)
        self.clear_scene()

    def streaming_segment(self) -> None:
        title = self.title("Acquisition and analysis overlap on the GPU", "04 / streaming path")
        stages = (
            VGroup(
                self.stage_box("CAMERA", "single frames", BLUE, 1.75),
                self.stage_box("DAQIRI", "receive + assemble", CYAN, 2.0),
                self.stage_box("BUCKET", "frame stacks", CREAM, 1.85),
                self.stage_box("CORRECT", "pedestal + mask", GREEN, 2.0),
                self.stage_box("COUNT", "electron events", ORANGE, 1.8),
                self.stage_box("ACCUMULATE", "spectra + maps", YELLOW, 2.2),
            )
            .arrange(RIGHT, buff=0.25)
            .move_to(UP * 0.75)
        )
        links = VGroup(
            *[
                Line(
                    stages[index].get_right(),
                    stages[index + 1].get_left(),
                    color="#58727d",
                    stroke_width=4,
                )
                for index in range(len(stages) - 1)
            ]
        )
        storage = self.stage_box("SELECTED HDF5", "validation bursts", MUTED, 2.55).move_to(
            RIGHT * 1.2 + DOWN * 1.2
        )
        storage_arrow = Arrow(
            stages[3].get_bottom(),
            storage.get_top(),
            buff=0.1,
            color=MUTED,
            stroke_width=2.4,
        )
        live = self.stage_box("LIVE VIEW", "spectrum + maps", CYAN, 2.55).move_to(
            RIGHT * 4.55 + DOWN * 1.2
        )
        live_arrow = Arrow(
            stages[5].get_bottom(), live.get_top(), buff=0.1, color=CYAN, stroke_width=2.4
        )
        self.play(FadeIn(title), FadeIn(stages), Create(links), run_time=0.9)
        moving = []
        dots = VGroup()
        for index in range(15):
            link = links[index % len(links)]
            dot = Dot(
                link.get_start(),
                radius=0.05,
                color=(BLUE, CYAN, GREEN, ORANGE)[index % 4],
            )
            dots.add(dot)
            moving.append(MoveAlongPath(dot, link, run_time=0.65))
        self.add(dots)
        self.play(LaggedStart(*moving, lag_ratio=0.08), run_time=2.1)
        self.play(
            GrowArrow(storage_arrow),
            FadeIn(storage),
            GrowArrow(live_arrow),
            FadeIn(live),
            run_time=0.8,
        )
        note = Text(
            "frame buckets accumulate while acquisition continues",
            color=CREAM,
            font_size=21,
        ).move_to(LEFT * 3.25 + DOWN * 1.25)
        self.play(FadeIn(note))
        caption = self.caption(
            "Single detector frames are assembled into GPU stacks before correction and counting."
        )
        self.play(FadeIn(caption))
        self.wait(2.8)
        self.clear_scene()

    def spectrum_segment(self) -> None:
        title = self.title(
            "At each point, corrected frames become an EELS spectrum", "05 / reduction"
        )
        spectrum = self.image_card(
            "analog_vs_counted_processing.png", width=11.65, shift=DOWN * 0.04
        )
        self.play(FadeIn(title), FadeIn(spectrum), run_time=0.9)
        edges = (
            VGroup(
                self.chip("Ti 456 eV", "edge fit", RED, 2.2),
                self.chip("O 532 eV", "edge fit", YELLOW, 2.2),
                self.chip("Mn 640 eV", "edge fit", GREEN, 2.2),
            )
            .arrange(RIGHT, buff=0.18)
            .scale(0.74)
            .to_edge(DOWN, buff=0.82)
        )
        self.play(FadeIn(edges), run_time=0.55)
        caption = self.caption(
            "One fitted edge amplitude per element becomes one map value at this (x, y)."
        )
        self.play(FadeIn(caption))
        self.wait(3.8)
        self.clear_scene()

    def live_maps_segment(self) -> None:
        title = self.title("Elemental maps update in probe-scan order", "06 / live reconstruction")
        filenames = ("final_haadf.png", "final_mn.png", "final_o.png", "final_ti.png")
        cards = Group(*[self.image_card(name, width=2.75) for name in filenames])
        cards.arrange(RIGHT, buff=0.23).move_to(DOWN * 0.02)
        covers_by_map = [self.map_pixel_covers(card[1]) for card in cards]
        all_covers = VGroup(
            *[cover_row for map_covers in covers_by_map for cover_row in map_covers]
        )
        scan_order = Text(
            "fast x: left to right  |  flyback  |  slow y: top to bottom",
            color=CYAN,
            font_size=15,
        ).move_to(UP * 2.22)
        track = Line(
            LEFT * 5.55 + DOWN * 2.55,
            RIGHT * 5.55 + DOWN * 2.55,
            color="#35505d",
            stroke_width=5,
        )
        progress = Line(track.get_start(), track.get_end(), color=CYAN, stroke_width=5)
        marker = Dot(track.get_start(), radius=0.085, color=ORANGE)
        start_label = Text("probe 1", color=MUTED, font_size=12).next_to(
            track.get_start(), DOWN, buff=0.08
        )
        end_label = Text("probe 256", color=MUTED, font_size=12).next_to(
            track.get_end(), DOWN, buff=0.08
        )
        self.play(
            FadeIn(title),
            FadeIn(scan_order),
            FadeIn(cards),
            FadeIn(all_covers),
            FadeIn(track),
            FadeIn(marker),
            FadeIn(start_label),
            FadeIn(end_label),
            run_time=0.8,
        )
        position_reveals = []
        for row in range(16):
            for column in range(16):
                position_reveals.append(
                    AnimationGroup(
                        *[
                            FadeOut(map_covers[row][column], run_time=0.18)
                            for map_covers in covers_by_map
                        ]
                    )
                )
        self.play(
            LaggedStart(*position_reveals, lag_ratio=0.2),
            MoveAlongPath(marker, progress, rate_func=linear),
            run_time=5.2,
        )
        badges = VGroup(
            Text("structure", color=CREAM, font_size=14, weight="BOLD"),
            Text("Mn edge", color=GREEN, font_size=14, weight="BOLD"),
            Text("O edge", color=YELLOW, font_size=14, weight="BOLD"),
            Text("Ti edge", color=RED, font_size=14, weight="BOLD"),
        )
        for badge, card in zip(badges, cards):
            badge.next_to(card, DOWN, buff=0.11)
        self.play(FadeIn(badges), run_time=0.5)
        caption = self.caption(
            "The reveal follows our x-fast raster order; elapsed animation time is schematic.",
            ORANGE,
        )
        self.play(FadeIn(caption))
        self.wait(2.4)
        self.clear_scene()

    def selection_segment(self) -> None:
        title = self.title("Select chemistry from the same registered acquisition", "07 / result")
        names = ("Structure", "Mn", "O", "Ti")
        filenames = ("final_haadf.png", "final_mn.png", "final_o.png", "final_ti.png")
        colors = (CREAM, GREEN, YELLOW, RED)
        details = (
            "HAADF high-angle intensity",
            "Mn edge amplitude near 640 eV",
            "O edge amplitude near 532 eV",
            "Ti edge amplitude near 456 eV",
        )
        current = self.image_card(filenames[0], width=5.7, shift=LEFT * 2.55 + DOWN * 0.03)
        selector = (
            VGroup(
                *[
                    self.stage_box(name.upper(), detail, color, 3.9, 0.92)
                    for name, detail, color in zip(names, details, colors)
                ]
            )
            .arrange(DOWN, buff=0.16)
            .move_to(RIGHT * 3.9 + DOWN * 0.02)
        )
        highlight = RoundedRectangle(
            width=4.08,
            height=1.06,
            corner_radius=0.13,
            fill_opacity=0,
            stroke_color=colors[0],
            stroke_width=3,
        ).move_to(selector[0])
        self.play(FadeIn(title), FadeIn(current), FadeIn(selector), FadeIn(highlight), run_time=0.9)
        for index in range(1, len(names)):
            next_card = self.image_card(
                filenames[index], width=5.7, shift=LEFT * 2.55 + DOWN * 0.03
            )
            self.play(
                FadeOut(current),
                FadeIn(next_card),
                highlight.animate.move_to(selector[index]).set_color(colors[index]),
                run_time=0.75,
            )
            current = next_card
            self.wait(0.9)
        caption = self.caption(
            "Element maps are fitted EELS-edge amplitudes - not painted atom labels."
        )
        self.play(FadeIn(caption))
        self.wait(2.7)
        self.clear_scene()

    def outro_segment(self) -> None:
        heading = Text(
            "From electrons to live elemental maps",
            color=CREAM,
            font_size=40,
            weight="BOLD",
        )
        stages = VGroup(
            self.stage_box("SPECIMEN", "LMTO lattice", YELLOW, 2.05),
            self.stage_box("CAMERA", "fast DOEELS", BLUE, 2.05),
            self.stage_box("DAQIRI", "packets to GPU", CYAN, 2.15),
            self.stage_box("ANALYSIS", "correct + fit", ORANGE, 2.15),
            self.stage_box("MAPS", "live feedback", GREEN, 2.05),
        ).arrange(RIGHT, buff=0.34)
        arrows = VGroup(
            *[
                Arrow(
                    stages[index].get_right(),
                    stages[index + 1].get_left(),
                    buff=0.05,
                    color="#58727d",
                    stroke_width=2.5,
                )
                for index in range(len(stages) - 1)
            ]
        )
        note = Text(
            "Advanced cameras create the signal. GPU-native DAQ makes it actionable sooner.",
            color=MUTED,
            font_size=19,
        )
        group = Group(heading, Group(stages, arrows), note).arrange(DOWN, buff=0.5)
        self.play(FadeIn(heading), run_time=0.6)
        self.play(FadeIn(stages), Create(arrows), run_time=1.0)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(3.0)
        self.play(FadeOut(group), run_time=0.6)


class MicroscopeScene(CombinedDemoScene):
    def construct(self):
        self.microscope_segment()


class PacketAssemblyScene(CombinedDemoScene):
    def construct(self):
        self.packet_segment()


class StreamingScene(CombinedDemoScene):
    def construct(self):
        self.streaming_segment()
        self.spectrum_segment()


class LiveMapsScene(CombinedDemoScene):
    def construct(self):
        self.live_maps_segment()


class ResultsScene(CombinedDemoScene):
    def construct(self):
        self.selection_segment()


class FullDemo(CombinedDemoScene):
    def construct(self):
        self.intro_segment()
        self.microscope_segment()
        self.scan_segment()
        self.packet_segment()
        self.streaming_segment()
        self.spectrum_segment()
        self.live_maps_segment()
        self.selection_segment()
        self.outro_segment()
