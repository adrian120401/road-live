"""CLI options kept outside the streaming video orchestration."""

import argparse
from pathlib import Path

from .config import (Config, RoadDamageConfig, LocationConfig, TrafficConfig,
                     CrosswalkConfig, ProximityConfig, MapConfig)


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Urban Vision: percepción urbana local.")
    parser.add_argument("--input", dest="input_path", required=True, type=Path)
    parser.add_argument("--output", dest="output_path", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--conf", type=float, default=.10)
    parser.add_argument("--display-conf",dest="display_confidence",type=float,default=Config.display_confidence,
                        help="Umbral visual general; no cambia asociación/analytics ni pozos")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    parser.add_argument("--processing-max-side",type=int,default=Config.processing_max_side,
                        help="Lado máximo para análisis/exportación (1920); 0 conserva resolución original")
    parser.add_argument("--prefetch-frames",type=int,default=Config.prefetch_frames,
                        help="Cola limitada de lectura anticipada (2); 0 usa lectura secuencial")
    parser.add_argument("--road-damage", action="store_true")
    parser.add_argument("--road-model", type=Path, default=RoadDamageConfig.model)
    parser.add_argument("--road-conf", type=float, default=.50, help="Mínimo obligatorio: 0.50")
    parser.add_argument("--road-interval", type=int, default=2)
    parser.add_argument("--road-roi", nargs=4, type=float, default=RoadDamageConfig.roi,
                        metavar=("LEFT","TOP","RIGHT","BOTTOM"))
    parser.add_argument("--debug-road", action="store_true")
    parser.add_argument("--road-area", nargs=8, type=float,
                        help="Zona de calzada: cuatro puntos X Y normalizados, en orden")
    parser.add_argument("--no-road-area", action="store_true", help="Desactivar filtro de veredas")
    parser.add_argument("--road-area-overlap", type=float, default=RoadDamageConfig.road_area_min_overlap,
                        help="Fracción mínima del box dentro de la calzada (0.60)")
    parser.add_argument("--no-road-evidence", action="store_true")
    parser.add_argument("--collect-road-candidates", action="store_true")
    parser.add_argument("--use-video-gps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-mock-route", action="store_true")
    parser.add_argument("--map-output", type=Path)
    parser.add_argument("--exiftool", type=Path)
    parser.add_argument("--enhanced-popups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--traffic-lights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crosswalks", action="store_true")
    parser.add_argument("--crosswalk-model", type=Path, default=CrosswalkConfig.model)
    parser.add_argument("--crosswalk-conf", type=float, default=CrosswalkConfig.confidence)
    parser.add_argument("--crosswalk-interval", type=int, default=CrosswalkConfig.frame_interval)
    parser.add_argument("--crosswalk-roi", nargs=4, type=float, default=CrosswalkConfig.roi)
    parser.add_argument("--proximity", action="store_true")
    parser.add_argument("--depth-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-model", type=Path, default=ProximityConfig.depth_model)
    parser.add_argument("--depth-interval", type=int, default=ProximityConfig.depth_interval)
    parser.add_argument("--depth-size", type=int, default=ProximityConfig.depth_image_size)
    parser.add_argument("--proximity-conf",type=float,default=ProximityConfig.candidate_confidence)
    parser.add_argument("--proximity-profile", type=Path, help="Thresholds medidos para esta cámara; sin perfil solo diagnóstico")
    parser.add_argument("--road-corridor", nargs=8, type=float,
                        help="Cuatro puntos X Y normalizados, ordenados alrededor del trapecio")
    parser.add_argument("--debug-scene", action="store_true")
    args = vars(parser.parse_args(argv))
    area_points = args.pop("road_area")
    disable_area = args.pop("no_road_area")
    if disable_area and area_points:
        parser.error("--road-area y --no-road-area son mutuamente excluyentes")
    road_area = None if disable_area else (tuple(zip(area_points[::2], area_points[1::2]))
                                          if area_points else RoadDamageConfig.road_area)
    road = RoadDamageConfig(enabled=args.pop("road_damage"),model=args.pop("road_model"),
        confidence=args.pop("road_conf"),frame_interval=args.pop("road_interval"),
        roi=tuple(args.pop("road_roi")),debug=args.pop("debug_road"),
        save_evidence=not args.pop("no_road_evidence"),collect_candidates=args.pop("collect_road_candidates"),
        road_area=road_area,road_area_min_overlap=args.pop("road_area_overlap"))
    location = LocationConfig(use_video_gps=args.pop("use_video_gps"),force_mock_route=args.pop("force_mock_route"),
                              map_output=args.pop("map_output"),exiftool=args.pop("exiftool"))
    crosswalk = CrosswalkConfig(enabled=args.pop("crosswalks"),model=args.pop("crosswalk_model"),
        confidence=args.pop("crosswalk_conf"),frame_interval=args.pop("crosswalk_interval"),roi=tuple(args.pop("crosswalk_roi")))
    traffic = TrafficConfig(args.pop("traffic_lights"),crosswalk)
    points = args.pop("road_corridor")
    corridor = tuple(zip(points[::2],points[1::2])) if points else ProximityConfig.corridor
    proximity = ProximityConfig(enabled=args.pop("proximity"),depth_enabled=args.pop("depth_enabled"),
        depth_model=args.pop("depth_model"),depth_interval=args.pop("depth_interval"),depth_image_size=args.pop("depth_size"),
        profile=args.pop("proximity_profile"),corridor=corridor,candidate_confidence=args.pop("proximity_conf"))
    map_config = MapConfig(args.pop("enhanced_popups"))
    if args["output_path"] is None:
        name = "urban_vision_v3_1_output.mp4" if crosswalk.enabled or proximity.enabled else (
            "urban_vision_v2_output.mp4" if road.enabled else "urban_vision_output.mp4")
        args["output_path"] = Path("outputs")/name
    return Config(**args,road_damage=road,location=location,traffic=traffic,proximity=proximity,map=map_config)
