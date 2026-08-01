from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from wta_env import WeaponTargetAssignmentEnv
from train_dqn import train, get_best_action, get_ranked_actions_per_weapon, save_model, load_model, verify_all_q_values_found




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
    unit_wei: str
    distance_from_cone_center_m: str
    distance_to_cone_boundary_m: str


class LocationAnalysis(BaseModel):
    abnormal: bool
    cluster_id: int
    distance_to_cluster_center: float
    threshold: float
    input: dict  # {"latitude": float, "longitude": float}
    cluster_center: dict  # {"latitude": float, "longitude": float}


class WeaponAnalysis(BaseModel):
    type: str
    frequency: float
    rare_threshold: float
    rare: bool
    unknown: bool


class ReasonDetail(BaseModel):
    code: str
    message: str


class AnomalyAnalysis(BaseModel):
    is_anomaly: bool
    model_anomaly: bool
    model_decision_score: Optional[float] = None
    reasons: List[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    service_error: Optional[str] = None
    location: Optional[LocationAnalysis] = None
    weapon: Optional[WeaponAnalysis] = None
    status: Optional[str] = None
    status_th: Optional[str] = None
    status_message_th: Optional[str] = None
    reasons_th: List[str] = Field(default_factory=list)
    reason_details: List[ReasonDetail] = Field(default_factory=list)
    explanation_th: Optional[str] = None


class ArtilleryEvent(BaseModel):
    event_id: str
    enemy_launch_point: EnemyLaunchPoint
    enemy_impact_point: EnemyImpactPoint
    candidate_cones: List[CandidateCone] = Field(default_factory=list)
    anomaly_analysis: AnomalyAnalysis


@app.post("/events/artillery")
async def receive_artillery_event(payload: ArtilleryEvent):
    # payload is already validated and parsed into typed objects here.
    # Add your processing logic (save to DB, trigger alerts, etc.) below.
    payload_dict = payload.model_dump()
    # print(payload_dict)


    # Weight Topic 
    # 

    NUM_WEAPONS = len(payload_dict["candidate_cones"])
    print("NUM_WEAPONS = ", NUM_WEAPONS)

    NUM_TARGETS = 1
    print("NUM_TARGETS = ", NUM_TARGETS)


    KILL_PROB = np.zeros((NUM_WEAPONS, NUM_TARGETS))
    # Row -> Weapon
    # Col -> Target
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

    # FInd KILL_PROB weight by range 
    KILL_PROB_range_weight = np.zeros((NUM_WEAPONS, NUM_TARGETS))
    for weapon in range(0,len(KILL_PROB)):
        for target in range(0 ,len(KILL_PROB[weapon, :])):
            weapon_range = float(payload_dict["candidate_cones"][weapon]["distance_from_cone_center_m"])
            Min_Active_Indirect_range = payload_dict["candidate_cones"][weapon]["cone_radii_m"][0]
            Max_Active_Indirect_range = payload_dict["candidate_cones"][weapon]["cone_radii_m"][1]
            Max_Extended_Indirect_range = payload_dict["candidate_cones"][weapon]["cone_radii_m"][2]

            # Calu Which range type Target in and set KILL_PROB_range_weight
            if weapon_range < Min_Active_Indirect_range:
                # If Target in Close range
                KILL_PROB_range_weight[weapon, target] = 0.0    # KILL_PROB   0 %
            elif weapon_range <= Max_Active_Indirect_range:
                # If Target in Active range
                KILL_PROB_range_weight[weapon, target] = 1.0    # KILL_PROB 100 %
            elif weapon_range < Max_Extended_Indirect_range:
                # If Target in Extend range
                KILL_PROB_range_weight[weapon, target] = 0.8    # KILL_PROB  80 %

            # Weighted Average by 1.00 (100 %)
            KILL_PROB = (
                        (KILL_PROB_range_weight * 1.00)     \
                                                            \
                        ) / 1.00

            print("Weapon :", weapon  , payload_dict["candidate_cones"][weapon]["cone_name"] ," --> TARGET" \
                  , target,  payload_dict["enemy_launch_point"]["name"] ,"\tKILL_PROB Rate : ",KILL_PROB[weapon, target])


            # print(KILL_PROB[weapon, target])
    print("KILL_PROB = ")
    print(KILL_PROB)


    WEAPON_VALUE = np.zeros(NUM_WEAPONS)
        # data                            [  1.0, 2.0, 3.0,   4.0]
        # scaled_data     WEAPON_VALUE =  [  1.0  4.0  7.0   10.0]
        # data                            [  1.0, 2.0, 3.0, 999.0]
        # scaled_data     WEAPON_VALUE =  [  1.0  1.0  1.0   10.0]
        # data                            [500.0 , 211.0, 341.0, 723.0]
        # scaled_data     WEAPON_VALUE =  [[ 6.1     1.0    3.3   10.0 ]]
        
    WEAPON_WEIGHT = []

    # Find Min / Max from unit_wei
    for weapon in range(0, NUM_WEAPONS):
        WEAPON_WEIGHT.append(round(float(payload_dict["candidate_cones"][weapon]["unit_wei"]), 2))

    WEAPON_WEIGHT_min = min(WEAPON_WEIGHT)
    WEAPON_WEIGHT_max = max(WEAPON_WEIGHT)

    # Min-Max scaling to range 1.0 to 10.0
    if WEAPON_WEIGHT_min == WEAPON_WEIGHT_max:
        # If min-max is equal mean all data is same.
        WEAPON_VALUE = np.ones(NUM_WEAPONS)

    else:
        # Else Find Min-Max Scaling to 1 to 10
        for weapon in range(0, NUM_WEAPONS):

            data = float(payload_dict["candidate_cones"][weapon]["unit_wei"])
            # Min-Max Scaling to 0 to 1
            WEAPON_VALUE[weapon] = ((data - WEAPON_WEIGHT_min) / (WEAPON_WEIGHT_max - WEAPON_WEIGHT_min))
            # Min-Max Scaling to 1 to 10
            WEAPON_VALUE[weapon] = ( WEAPON_VALUE[weapon] * (10 - 1) ) + 1
            # round to 1.0
            WEAPON_VALUE[weapon] = round(float(WEAPON_VALUE[weapon]), 1)
    print("WEAPON_VALUE is Min-Max Scaling to 1.0 to 10.0 - Scaling by Min Max of unit_wei")
    print("WEAPON_VALUE = ", WEAPON_VALUE)
    print("    unit_wei = ", WEAPON_WEIGHT)


    # TARGET_VALUE = [7.]
    # TARGET_VALUE set by number of reasons
    TARGET_VALUE = np.array( [1.0] )
    if payload_dict["anomaly_analysis"]["is_anomaly"] == True:
        # IF Anomaly set to number of reasons.
        TARGET_VALUE[0] = len(payload_dict["anomaly_analysis"]["reasons"])
    else:
        # IF not Anomaly set to 1.0
        TARGET_VALUE = np.array( [1.0] )

    print("TARGET_VALUE = ", TARGET_VALUE, "(Not effect if target is only one.)")


    # --------------------------------------------------------------------------

    MODEL_PATH = f"wta_dqn_{NUM_WEAPONS}w_{NUM_TARGETS}t.pt"
    # 1. Get a trained Q-network: load from disk if available, else train one.
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained model from {MODEL_PATH} ...")
        q_net = load_model(MODEL_PATH)
    else:
        print("No saved model found, training a small one (this may take a bit) ...")
        q_net, _ = train(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS, num_episodes=1000)
        save_model(q_net, MODEL_PATH)

    # 2. Set up the environment and load the (random or custom) scenario above
    #    as the current state.
    env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    obs, info = env.reset(options={
        "kill_prob": KILL_PROB,
        "target_value": TARGET_VALUE,
        "weapon_value": WEAPON_VALUE,
    })

    print("\nStarting a new episode. Kill probability matrix (weapons x targets):")
    print(env.kill_prob.round(2))
    print("Target values:", env.target_value.round(2))
    print("Weapon values:", env.weapon_value.round(2))

    # 3. Repeatedly ask the model for the best action given the current state.
    step = 0
    done = False
    total_reward = 0.0
    prediction_result = []

    while not done:
        action_mask = info["action_mask"]

        # Confirm the network produced a finite Q-value for EVERY one of
        # num_weapons * num_targets actions before trusting its argmax --
        # i.e. make sure all Q values are actually found, nothing missing.
        q_check = verify_all_q_values_found(q_net, obs, env.num_weapons, env.num_targets)
        if not q_check["complete"]:
            print(f"  [WARNING] Step {step + 1}: Q-value grid incomplete! "
                  f"expected {q_check['expected_count']}, got {q_check['actual_count']}, "
                  f"bad indices: {q_check['missing_or_bad_indices']}")
        else:
            print(f"--- Step {step + 1}: verified all {q_check['expected_count']} Q-values present and finite ---")

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

        prediction_result.append({
        "weapon": weapon_idx,
        "target": target_idx,
        "reward": round(float(reward), 4),
        "objective_after": round(info["objective_value"], 4),
        })

        print(f"Step {step}: assign weapon {weapon_idx} -> target {target_idx} "
              f"| reward={reward:.3f} | objective (lower=better)={info['objective_value']:.3f}")


        if step == 1:
            step_1_action_reward = np.round(all_rewards, 2)
            step_1_top_unique = top_unique

            print(ranked)

    print(f"\nEpisode finished in {step} steps. "
          f"Total expected damage dealt: {total_reward:.3f}")
    print(f"Final expected surviving target value: {info['objective_value']:.3f}")
    print(f"Total weapon value used: {info['total_weapon_value_used']:.3f}")

    # 4. Compare against a pure greedy baseline on the EXACT same scenario --
    #    greedy always takes the single locally-best action (no lookahead),
    #    while the DQN can learn to hold weapons back for a better long-term
    #    outcome. Lower final_objective is better.
    # greedy_env = WeaponTargetAssignmentEnv(num_weapons=NUM_WEAPONS, num_targets=NUM_TARGETS)
    # greedy_env.reset(options={
    #     "kill_prob": KILL_PROB,
    #     "target_value": TARGET_VALUE,
    #     "weapon_value": WEAPON_VALUE,
    # })
    # greedy_result = greedy_env.greedy_solve()

    # print("\n=== Greedy baseline on the same scenario (no lookahead) ===")
    # for w, t, r in greedy_result["assignments"]:
    #     print(f"  weapon {w} -> target {t}: reward={r:.3f}")
    # print(f"  total_reward={greedy_result['total_reward']}, "
    #       f"final_objective={greedy_result['final_objective']} (lower is better)")
    # print(f"\nDQN final_objective={info['objective_value']:.4f} vs "
    #       f"Greedy final_objective={greedy_result['final_objective']} (lower is better)")

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

    print("top_unique        :" , step_1_top_unique)
    print("prediction_result :" , prediction_result)

    for i in range(0, len(prediction_result)):
        prediction_result[i] = (prediction_result[i]['weapon'], prediction_result[i]['target'], prediction_result[i]['reward'])
    prediction_result = sorted(prediction_result, key=lambda x: x[-1], reverse=True)
    
    # top_return = step_1_action_reward
    top_return = step_1_top_unique
    # top_return = prediction_result

    # Data from      Q Learning
    print("top_unique        :" , step_1_top_unique)
    # Data from Deep Q Learning
    print("prediction_result :" , prediction_result)

    if len(top_return) > 3:
        top_return = top_return[0:3]
    else:
        pass
    # print(top_return)

    top_result = []
    # print(top_result)
    
    my_rank = 1
    for w, t, r in top_return:
        print(f"  weapon {w} -> target {t}: reward={r:.2f}")

        top_result.append(  {   
                                "weapon_rank"   : my_rank,
                                "weapon_id"     : w,
                                "weapon"        : payload_dict["candidate_cones"][w],
                                "target_id"     : t,
                                "target"        : payload_dict["enemy_launch_point"],

                                "reward"        : r
                         }  )
        my_rank = my_rank + 1

    response_data["result"] = top_result

    # print(response_data)

    return response_data


@app.get("/health")
async def health():
    return {"status": "ok"}
