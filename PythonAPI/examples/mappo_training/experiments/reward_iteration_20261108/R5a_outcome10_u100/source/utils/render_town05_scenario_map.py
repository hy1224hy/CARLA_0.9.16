"""Render the frozen Town05 v1 training scenarios as a project artifact.

This script is intentionally independent from the pygame training visualizers.
It builds the road background from Town05.xodr and overlays the scenario JSON
definitions, so it can be used even when no CARLA server is running.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import carla
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[4]
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = PROJECT_DIR / "data" / "training_scenarios_town05_v1"
DEFAULT_XODR = ROOT / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / "Town05.xodr"
DEFAULT_OUTPUT = PROJECT_DIR / "docs" / "town05_training_scenarios_marked.png"

FAMILY_META = {
    "parallel_stationary": ("D1  Parallel stationary", "Zero-speed start; basic lane keeping and launch"),
    "parallel_rolling": ("D2  Parallel rolling", "Equal-speed rolling start; side-by-side interaction"),
    "stagger_rear_closing": ("D3  Rear closing", "Staggered start; rear vehicle closes the gap"),
    "required_merge": ("D4  Required merge", "One agent must enter the other agent's target lane"),
}
FAMILY_ORDER = list(FAMILY_META)

COLORS = {
    "background": "#f5f7fb",
    "panel": "#ffffff",
    "border": "#cfd7e6",
    "road": "#d2d8e2",
    "road_train": "#7c93ba",
    "road_holdout": "#a98dcb",
    "text": "#172033",
    "muted": "#68758b",
    "agent0": "#1976d2",
    "agent1": "#ef6c00",
    "goal": "#2e9d57",
    "route": "#4b5f7d",
    "merge": "#c026d3",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(38, bold=True)
FONT_SUBTITLE = load_font(20)
FONT_PANEL = load_font(25, bold=True)
FONT_BODY = load_font(17)
FONT_LABEL = load_font(17, bold=True)
FONT_SMALL = load_font(14)


def load_scenarios(directory: Path) -> list[dict]:
    scenarios = []
    for path in sorted(directory.glob("scenario_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            scenario = json.load(handle)
        scenario["_path"] = path
        scenario["_id"] = path.name.split("_")[1]
        scenarios.append(scenario)
    if len(scenarios) != 16:
        raise RuntimeError(f"Expected 16 Town05 scenarios, found {len(scenarios)} in {directory}")
    return scenarios


def load_lane_lines(xodr_path: Path) -> dict[tuple[int, int, int], list[tuple[float, float]]]:
    carla_map = carla.Map("Town05", xodr_path.read_text(encoding="utf-8"))
    grouped: dict[tuple[int, int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for waypoint in carla_map.generate_waypoints(1.5):
        if waypoint.lane_type != carla.LaneType.Driving:
            continue
        location = waypoint.transform.location
        grouped[(waypoint.road_id, waypoint.section_id, waypoint.lane_id)].append(
            (waypoint.s, location.x, location.y)
        )
    return {
        key: [(x, y) for _, x, y in sorted(points)]
        for key, points in grouped.items()
        if len(points) >= 2
    }


def data_bounds(points: Iterable[tuple[float, float]], padding: float = 0.0) -> tuple[float, float, float, float]:
    points = list(points)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding


def make_transform(
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
    inner_padding: int = 20,
):
    min_x, max_x, min_y, max_y = bounds
    left, top, right, bottom = rect
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    scale = min(
        (right - left - 2 * inner_padding) / width,
        (bottom - top - 2 * inner_padding) / height,
    )
    draw_width = width * scale
    draw_height = height * scale
    x0 = left + (right - left - draw_width) / 2
    y0 = top + (bottom - top - draw_height) / 2

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return x0 + (x - min_x) * scale, y0 + draw_height - (y - min_y) * scale

    return transform


def draw_lane_background(
    draw: ImageDraw.ImageDraw,
    lane_lines: dict[tuple[int, int, int], list[tuple[float, float]]],
    transform,
    clip_rect: tuple[int, int, int, int],
    base_width: int,
) -> None:
    left, top, right, bottom = clip_rect
    for (road_id, _, _), points in lane_lines.items():
        screen = [transform(point) for point in points]
        color = COLORS["road"]
        width = base_width
        if road_id in {44, 3}:
            color = COLORS["road_train"]
            width = base_width + 2
        elif road_id in {43, 6}:
            color = COLORS["road_holdout"]
            width = base_width + 1
        # Pillow does not clip paths to a panel.  Drawing only segments whose
        # endpoints are in the map rectangle prevents a lane from bleeding
        # through adjacent panels; the 1.5 m waypoint spacing keeps edges neat.
        for point_a, point_b in zip(screen, screen[1:]):
            if all(
                left <= x <= right and top <= y <= bottom
                for x, y in (point_a, point_b)
            ):
                draw.line([point_a, point_b], fill=color, width=width)


def oriented_box(center: tuple[float, float], yaw: float, half_length: float, half_width: float):
    angle = math.radians(yaw)
    forward = (math.cos(angle), math.sin(angle))
    left = (-math.sin(angle), math.cos(angle))
    return [
        (
            center[0] + forward[0] * sx * half_length + left[0] * sy * half_width,
            center[1] + forward[1] * sx * half_length + left[1] * sy * half_width,
        )
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ]


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color: str, width: int = 3) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 11
    wing_a = (
        end[0] - size * math.cos(angle - math.pi / 6),
        end[1] - size * math.sin(angle - math.pi / 6),
    )
    wing_b = (
        end[0] - size * math.cos(angle + math.pi / 6),
        end[1] - size * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, wing_a, wing_b], fill=color)


def draw_agent(draw: ImageDraw.ImageDraw, point, color: str, label: str) -> None:
    radius = 8
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
    draw.text((x + 10, y - 12), label, fill=color, font=FONT_SMALL, stroke_width=2, stroke_fill="white")


def draw_scenario_overlay(draw: ImageDraw.ImageDraw, scenario: dict, transform) -> None:
    agents = scenario["agents"]
    world_a0 = (agents["agent0"]["x"], agents["agent0"]["y"])
    world_a1 = (agents["agent1"]["x"], agents["agent1"]["y"])
    goal = scenario["goal_zone"]
    world_goal = (goal["x"], goal["y"])
    screen_a0 = transform(world_a0)
    screen_a1 = transform(world_a1)
    screen_goal = transform(world_goal)
    midpoint = ((screen_a0[0] + screen_a1[0]) / 2, (screen_a0[1] + screen_a1[1]) / 2)

    draw_arrow(draw, midpoint, screen_goal, COLORS["route"], width=2)
    goal_polygon = [
        transform(point)
        for point in oriented_box(
            world_goal,
            goal.get("yaw", 0.0),
            goal.get("half_length", 3.0),
            goal.get("half_width", 3.4),
        )
    ]
    draw.polygon(goal_polygon, fill="#dff3e6", outline=COLORS["goal"], width=3)
    draw_agent(draw, screen_a0, COLORS["agent0"], "A0")
    draw_agent(draw, screen_a1, COLORS["agent1"], "A1")

    if scenario["family"] == "required_merge":
        merge_agent = "agent0" if "agent0_merge" in scenario.get("tags", []) else "agent1"
        merge_world = (agents[merge_agent]["x"], agents[merge_agent]["y"])
        draw_arrow(draw, transform(merge_world), screen_goal, COLORS["merge"], width=3)

    label_x, label_y = midpoint
    label = f"S{scenario['_id']}"
    draw.rounded_rectangle(
        (label_x - 22, label_y + 13, label_x + 22, label_y + 38),
        radius=7,
        fill="#ffffff",
        outline=COLORS["border"],
        width=1,
    )
    draw.text((label_x, label_y + 25), label, fill=COLORS["text"], font=FONT_LABEL, anchor="mm")


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    family: str,
    scenarios: list[dict],
    lane_lines: dict[tuple[int, int, int], list[tuple[float, float]]],
    training_bounds: tuple[float, float, float, float],
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=18, fill=COLORS["panel"], outline=COLORS["border"], width=2)
    title, subtitle = FAMILY_META[family]
    draw.text((left + 22, top + 18), title, fill=COLORS["text"], font=FONT_PANEL)
    draw.text((left + 22, top + 55), subtitle, fill=COLORS["muted"], font=FONT_BODY)
    map_rect = (left + 12, top + 88, right - 12, bottom - 12)
    transform = make_transform(training_bounds, map_rect, inner_padding=14)
    draw_lane_background(draw, lane_lines, transform, map_rect, base_width=2)
    for scenario in scenarios:
        draw_scenario_overlay(draw, scenario, transform)


def draw_overview(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    scenarios: list[dict],
    lane_lines: dict[tuple[int, int, int], list[tuple[float, float]]],
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=18, fill=COLORS["panel"], outline=COLORS["border"], width=2)
    draw.text((left + 22, top + 18), "Town05 road network overview", fill=COLORS["text"], font=FONT_PANEL)
    draw.text(
        (left + 22, top + 55),
        "Blue: training Roads 44/3    Purple: held-out Roads 43/6    Labels show the four directional scenario groups",
        fill=COLORS["muted"],
        font=FONT_BODY,
    )
    map_rect = (left + 12, top + 86, right - 12, bottom - 12)
    all_points = [point for points in lane_lines.values() for point in points]
    transform = make_transform(data_bounds(all_points, padding=8.0), map_rect, inner_padding=10)
    draw_lane_background(draw, lane_lines, transform, map_rect, base_width=2)

    by_direction: dict[str, list[dict]] = defaultdict(list)
    for scenario in scenarios:
        direction = next(tag for tag in scenario.get("tags", []) if tag.endswith("bound"))
        by_direction[direction].append(scenario)
    for direction, group in by_direction.items():
        representative = sorted(group, key=lambda item: item["_id"])[0]
        agents = representative["agents"]
        start = (
            (agents["agent0"]["x"] + agents["agent1"]["x"]) / 2,
            (agents["agent0"]["y"] + agents["agent1"]["y"]) / 2,
        )
        goal = representative["goal_zone"]
        screen_start = transform(start)
        screen_goal = transform((goal["x"], goal["y"]))
        draw_arrow(draw, screen_start, screen_goal, COLORS["route"], width=4)
        ids = "/".join(f"S{item['_id']}" for item in sorted(group, key=lambda item: item["_id"]))
        label = f"{direction.replace('bound', '').title()}  {ids}"
        x, y = screen_start
        box = draw.textbbox((0, 0), label, font=FONT_LABEL)
        label_width = box[2] - box[0]
        draw.rounded_rectangle(
            (x - 6, y - 36, x + label_width + 10, y - 8),
            radius=7,
            fill="#ffffff",
            outline=COLORS["border"],
            width=1,
        )
        draw.text((x + 2, y - 33), label, fill=COLORS["text"], font=FONT_LABEL)

    for road_id, label in [(44, "Road 44"), (3, "Road 3"), (43, "Road 43 (hold-out)"), (6, "Road 6 (hold-out)")]:
        points = [point for (rid, _, _), line in lane_lines.items() if rid == road_id for point in line]
        if not points:
            continue
        cx = sum(point[0] for point in points) / len(points)
        cy = sum(point[1] for point in points) / len(points)
        sx, sy = transform((cx, cy))
        draw.text((sx, sy), label, fill=COLORS["text"], font=FONT_SMALL, anchor="mm", stroke_width=3, stroke_fill="white")


def render(scenario_dir: Path, xodr_path: Path, output: Path) -> None:
    scenarios = load_scenarios(scenario_dir)
    lane_lines = load_lane_lines(xodr_path)
    training_points = []
    for scenario in scenarios:
        for agent in scenario["agents"].values():
            training_points.append((agent["x"], agent["y"]))
        goal = scenario["goal_zone"]
        training_points.append((goal["x"], goal["y"]))
    training_bounds = data_bounds(training_points, padding=14.0)

    image = Image.new("RGB", (2000, 1880), COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "Town05 · 16 frozen training scenarios", fill=COLORS["text"], font=FONT_TITLE)
    draw.text(
        (70, 87),
        "Source: Town05.xodr + data/training_scenarios_town05_v1 (not the legacy pygame visualizer)",
        fill=COLORS["muted"],
        font=FONT_SUBTITLE,
    )
    draw_overview(draw, (70, 130, 1930, 745), scenarios, lane_lines)

    grouped = defaultdict(list)
    for scenario in scenarios:
        grouped[scenario["family"]].append(scenario)
    panels = [
        (70, 785, 980, 1290),
        (1020, 785, 1930, 1290),
        (70, 1330, 980, 1835),
        (1020, 1330, 1930, 1835),
    ]
    for family, rect in zip(FAMILY_ORDER, panels):
        draw_panel(
            draw,
            rect,
            family,
            sorted(grouped[family], key=lambda item: item["_id"]),
            lane_lines,
            training_bounds,
        )

    legend_y = 1852
    items = [
        (COLORS["agent0"], "Agent 0 start"),
        (COLORS["agent1"], "Agent 1 start"),
        (COLORS["goal"], "Goal zone"),
        (COLORS["merge"], "Required merge path"),
    ]
    x = 590
    for color, label in items:
        draw.ellipse((x, legend_y, x + 14, legend_y + 14), fill=color)
        draw.text((x + 21, legend_y - 5), label, fill=COLORS["muted"], font=FONT_SMALL)
        x += 205

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    print(f"Rendered {len(scenarios)} scenarios to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--xodr", type=Path, default=DEFAULT_XODR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.scenario_dir.resolve(), args.xodr.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
