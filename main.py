from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pprint

import json

app = FastAPI()

class EventIgnore_EventId(BaseModel):
    event_id: List[str]

class LaunchPoint(BaseModel):
    id: str
    name: str
    feature_key: str
    event_time: str
    timestamp: float
    lat: float
    lon: float
    height: Optional[float] = None
    heading: float
    weapon_type: str
    source: str
    zone: Optional[str] = None
    east: Optional[str] = None
    north: Optional[str] = None
    dis: Optional[float] = None
    firV: Optional[float] = None
    firEL: Optional[float] = None


class ImpactPoint(BaseModel):
    id: str
    feature_key: str
    event_time: str
    timestamp: float
    lat: float
    lon: float
    height: Optional[float] = None
    heading: float
    weapon_type: str
    source: str


class CandidateCone(BaseModel):
    cone_id: int
    overlay_id: int
    feature_id: str
    cone_name: str
    plan_id: int
    mission_id: int
    unit_id: int
    center_lat: float
    center_lon: float
    cone_angle: str
    cone_heading: str
    cone_angle_left: str
    cone_angle_right: str
    cone_radii_m: List[float] = []
    distance_from_cone_center_m: str
    distance_to_cone_boundary_m: str


class EnemyEvent(BaseModel):
    event_id: str
    enemy_launch_point: LaunchPoint
    enemy_impact_point: ImpactPoint
    candidate_cones: List[CandidateCone]



@app.post("/ignore-event-id")
async def receive_event(event: EventIgnore_EventId):
    data: dict = event.model_dump()
    file_path = "data/ignore-event-id.json"

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data_list = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data_list = []

    data_list.append(data)

    # get data -> if data not in file : add to file :else do nothing 

    # data_list = data

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data_list, file, indent=4)

    return {"status": "ok", "event_id": data["event_id"]}


@app.post("/event-post-data")
async def receive_event(event: EnemyEvent):
    data: dict = event.model_dump()  # full payload as a plain dict

    print("Event ID:", data["event_id"])
    print("Launch coords:", data["enemy_launch_point"]["lat"], data["enemy_launch_point"]["lon"])
    print("Cones received:", len(data["candidate_cones"]))

    # pprint.pprint(data)



    # Add json to file 

    file_path = "data/enemy_point.json"

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data_list = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data_list = []

    data_list.append(data)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data_list, file, indent=4)

    return {"status": "ok", "event_id": data["event_id"]}