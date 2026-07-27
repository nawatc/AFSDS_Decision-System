from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field



import os
import numpy as np

from wta_env import WeaponTargetAssignmentEnv
from train_dqn import train, get_best_action, get_ranked_actions_per_weapon, save_model, load_model




app = FastAPI(title="Artillery Event API")


class EnemyLaunchPoint(BaseModel):
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
    zone: str
    east: str
    north: str
    dis: float
    firV: float
    firEL: float


class EnemyImpactPoint(BaseModel):
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
    cone_radii_m: List[float] = Field(default_factory=list)
    distance_from_cone_center_m: str
    distance_to_cone_boundary_m: str


class ArtilleryEvent(BaseModel):
    event_id: str
    enemy_launch_point: EnemyLaunchPoint
    enemy_impact_point: EnemyImpactPoint
    candidate_cones: List[CandidateCone] = Field(default_factory=list)


@app.post("/events/artillery")
async def receive_artillery_event(payload: ArtilleryEvent):
    # payload is already validated and parsed into typed objects here.
    # Add your processing logic (save to DB, trigger alerts, etc.) below.
    payload_dict = payload.model_dump()

    
    NUM_WEAPONS = len(payload_dict["candidate_cones"])
    # print(len(payload_dict["candidate_cones"]))
    print("NUM_WEAPONS = ", NUM_WEAPONS)

    NUM_TARGETS = 1
    # print(payload_dict["enemy_launch_point"])
    print("NUM_TARGETS = ", NUM_TARGETS)

    #     T1 T2 T3 ...
    # W1
    # W2
    # W3
    # ...

    # KILL_PROB = np.array([
    #     [0.8, 0.3, 0.5, 0.2],
    #     [0.4, 0.7, 0.6, 0.3],
    #     [0.6, 0.5, 0.9, 0.4],
    #     [0.3, 0.6, 0.4, 0.8],
    #     [0.7, 0.4, 0.3, 0.5],
    #     [0.5, 0.5, 0.6, 0.6],
    # ])

    # KILL_PROB = np.array( [   [1, 2],
    #                           [3, 4]]
    #                                     )
    KILL_PROB = np.zeros((NUM_WEAPONS, NUM_TARGETS))
    for weapon in range(0,len(KILL_PROB)):

        for target in range(0 ,len(KILL_PROB[weapon, :])):

            # Calulate Distance and Rate Probability of Kill (KILL_PROB)
            # FRIEND_WEAPON_lat = 1
            # FRIEND_WEAPON_lon = 1
            
            # TARGET_lat = 1
            # TARGET_lon = 1
            # 

            # Circular Error Probable (CEP) of M101A2
            # Ref. https://en.wikipedia.org/wiki/Circular_error_probable
            # Use Estimate Value
            # print(type(payload_dict["candidate_cones"][weapon]["distance_from_cone_center_m"]))



            # Rate Probability of Kill (KILL_PROB)
            range_est = float(payload_dict["candidate_cones"][weapon]["distance_from_cone_center_m"])
            # print(payload_dict["candidate_cones"][weapon]["distance_from_cone_center_m"])
            KILL_PROB_est = 0.4

            if payload_dict["candidate_cones"][weapon]["cone_name"] == "ปืน 105":
                if range_est < 5000:
                    KILL_PROB_est = 0.9
                elif range_est >= 5000 and range_est < 7500:
                    KILL_PROB_est = 0.8
                elif range_est >= 7500 and range_est < 10000:
                    KILL_PROB_est = 0.75
                elif range_est >= 10000:
                    KILL_PROB_est = 0.7
            if payload_dict["candidate_cones"][weapon]["cone_name"] == "ปืน DTI":
                if range_est < 5000:
                    KILL_PROB_est = 0.6
                elif range_est >= 5000 and range_est < 7500:
                    KILL_PROB_est = 0.5
                elif range_est >= 7500 and range_est < 10000:
                    KILL_PROB_est = 0.45
                elif range_est >= 10000:
                    KILL_PROB_est = 0.4
            
            KILL_PROB[weapon, target] = KILL_PROB_est

            # print(payload_dict["candidate_cones"][weapon]["cone_name"], "\tKILL_PROB\t", KILL_PROB[weapon, target])
            # print(payload_dict["candidate_cones"][weapon]["cone_name"])
            # print(KILL_PROB[weapon, target])
            print("Weapon :", weapon  , payload_dict["candidate_cones"][weapon]["cone_name"] ," --> TARGET" \
                  , target,  payload_dict["enemy_launch_point"]["name"] ,"\tKILL_PROB Rate : ",KILL_PROB[weapon, target])


            # print(KILL_PROB[weapon, target])
    print("KILL_PROB = ")
    print(KILL_PROB)

    # TARGET_VALUE = np.array(  [2.0, 8.0, 5.0, 9.0]    )
    # Range 0.1 to 10.0
    TARGET_VALUE = np.empty(1)
    TARGET_VALUE[0] = 10.0

    # for target in TARGET_VALUE:
    #     print(target)
    print("TARGET_VALUE = ", TARGET_VALUE)



    # --------------------------------------------------------------------------

    # Model filename is tied to the problem size, so changing NUM_WEAPONS/NUM_TARGETS
    # above always trains/loads a matching checkpoint instead of accidentally
    # loading a model shaped for a different size.
    MODEL_PATH = f"wta_dqn_{NUM_WEAPONS}w_{NUM_TARGETS}t.pt"

    assert KILL_PROB.shape == (NUM_WEAPONS, NUM_TARGETS), (
        f"KILL_PROB shape {KILL_PROB.shape} must match "
        f"(NUM_WEAPONS, NUM_TARGETS) = ({NUM_WEAPONS}, {NUM_TARGETS})"
    )
    assert TARGET_VALUE.shape == (NUM_TARGETS,), (
        f"TARGET_VALUE shape {TARGET_VALUE.shape} must match (NUM_TARGETS,) = ({NUM_TARGETS},)"
    )

    # 1. Get a trained Q-network: load from disk if available, else train one.
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained model from {MODEL_PATH} ...")
        q_net = load_model(MODEL_PATH)
    else:
        print("No saved model found, training a small one (this may take a bit) ...")
        q_net, _ = train(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS, num_episodes=2000)
        save_model(q_net, MODEL_PATH)

    # 2. Set up the environment and load the (random or custom) scenario above
    #    as the current state.
    env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    obs, info = env.reset(options={"kill_prob": KILL_PROB, "target_value": TARGET_VALUE})

    print("\nStarting a new episode. Kill probability matrix (weapons x targets):")
    print(env.kill_prob.round(2))
    print("Target values:", env.target_value.round(2))

    # 3. Repeatedly ask the model for the best action given the current state.
    step = 0
    done = False
    total_reward = 0.0
    prediction_result = []
    
    while not done:
        action_mask = info["action_mask"]

        # Ground-truth reward the env would give for EVERY possible action
        # right now (not a network estimate -- computed from the reward formula).
        all_rewards = env.get_all_action_rewards()
        print(f"\n--- Step {step + 1}: all action rewards (weapons x targets) ---")
        print(np.round(all_rewards, 2))

        # Same rewards, but sorted best-first as a flat ranked list.
        top_rewards = env.get_ranked_action_rewards(top_k=5)
        print(f"--- Step {step + 1}: top actions by actual reward ---")
        for w, t, r in top_rewards:
            print(f"  weapon {w} -> target {t}: reward={r:.2f}")
        print(top_rewards)

        # Same, but excluding targets that already have a weapon assigned to
        # them -- i.e. results where no target repeats.
        top_unique = env.get_ranked_action_rewards(top_k=5, unique_targets=True)
        print(f"--- Step {step + 1}: top actions, no repeated targets ---")
        if top_unique:
            for w, t, r in top_unique:
                print(f"  weapon {w} -> target {t}: reward={r:.2f}")
        else:
            print("  (every target already has a weapon assigned)")

        # For each unassigned weapon, show its best / 2nd-best / 3rd-best
        # target choice by Q-value (not just the single greedy pick).
        ranked = get_ranked_actions_per_weapon(q_net, obs, action_mask, env.num_targets, top_k=3)
        print(f"--- Step {step + 1} Q-value rankings ---")
        for weapon_idx, choices in ranked.items():
            choice_str = ", ".join(
                f"#{rank+1} target {t} (Q={q:.2f})" for rank, (t, q) in enumerate(choices)
            )
            print(f"  Weapon {weapon_idx}: {choice_str}")

        # <<< This is the core call: get the best action from the current state >>>
        action = get_best_action(q_net, obs, action_mask)

        weapon_idx, target_idx = divmod(action, env.num_targets)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        prediction_result.append({           # <-- ADD THIS
        "weapon": weapon_idx,
        "target": target_idx,
        "reward": round(float(reward), 4),
        "objective_after": round(info["objective_value"], 4),
        })

        print(f"Step {step}: assign weapon {weapon_idx} -> target {target_idx} "
              f"| reward={reward:.3f} | objective (lower=better)={info['objective_value']:.3f}")

        
        


    print(f"\nEpisode finished in {step} steps. "
          f"Total expected damage dealt: {total_reward:.3f}")
    print(f"Final expected surviving target value: {info['objective_value']:.3f}")


    """
    return {
        "status": "received",
        "event_id": payload.event_id,
        "launch_point": {
            "id": payload.enemy_launch_point.id,
            "lat": payload.enemy_launch_point.lat,
            "lon": payload.enemy_launch_point.lon,
        },
        "impact_point": {
            "id": payload.enemy_impact_point.id,
            "lat": payload.enemy_impact_point.lat,
            "lon": payload.enemy_impact_point.lon,
        },
        "candidate_cone_count": len(payload.candidate_cones),
        "result": {
        
        }
    }
    
    """
    print(prediction_result)

    response_data = {
        "status": "received",
        "event_id": payload.event_id,
        "launch_point": {
            "id": payload.enemy_launch_point.id,
            "lat": payload.enemy_launch_point.lat,
            "lon": payload.enemy_launch_point.lon,
        },
        "impact_point": {
            "id": payload.enemy_impact_point.id,
            "lat": payload.enemy_impact_point.lat,
            "lon": payload.enemy_impact_point.lon,
        },
        "candidate_cone_count": len(payload.candidate_cones),
        "event_id": payload.event_id,
    }
    if top_unique:
        top_return = top_unique
    elif top_rewards:
        top_return = top_rewards
    else:
        top_return = []

    top_result = []

    for w, t, r in top_return:
        print(f"  weapon {w} -> target {t}: reward={r:.2f}")

        top_result.append(  {   
                                "weapon_id" : w,
                                "weapon"    : payload_dict["candidate_cones"][w],
                                "target_id" : t,
                                "target"    : payload_dict["enemy_launch_point"],
                                "reward"    : r
                         }  )
    result_dict = {}

    response_data["result"] = top_result

    return response_data


@app.get("/health")
async def health():
    return {"status": "ok"}
