"""Strategy: full decision engine implementing 策略设计文档 L2–L8.

Priority order per frame (策略文档 §14 伪代码):
  P0  Stable online, every-frame heartbeat, zero illegal actions
  P1  Must deliver (goodFruit>0, freshness>0, verified, at terminal)
  P2  Task score ≥ 90
  P3  Deliver early (time score)
  P4  Preserve good fruit & freshness
  P5  Moderate combat (guard/break/squad) without sacrificing P1–P4
"""

from __future__ import annotations

import logging
from typing import Any

from lychee_client.map_graph import MapGraph, ROUTE_FRESHNESS_LOSS
from lychee_client.state import (
    can_move, can_act, get_current_node_id, needs_processing,
    is_delivered, is_retired, is_verified, is_at_node, is_in_passive_state,
    is_in_limited_state,
    find_available_resources, find_task_at_node,
    get_good_fruit, get_bad_fruit, get_freshness,
    get_player_resources, has_resource, get_squad_count,
    get_action_points, get_task_score, get_blocked_nodes,
    classify_opponent_mode, get_team_id, get_task_template_id,
    is_verify_process, is_enemy_guard, guard_is_active, node_has_obstacle,
    is_key_pass_node, find_node_by_id,
    TASK_SCORE_TARGET, TASK_SCORE_STRETCH, MAX_TASK_DETOUR_COST,
    ICE_BOX_FRESHNESS_THRESHOLD, RUSH_PROTECT_FRESHNESS, INTEL_MAX_DISTANCE,
    RESOURCE_CLAIM_PRIORITY, TASK_PRIORITY,
)
from lychee_client.decision import (
    make_action, make_move_action, make_wait_action,
    make_process_action, make_dock_action, make_verify_gate_action,
    make_empty_action, make_window_card_action,
    make_claim_resource_action, make_claim_task_action,
    make_deliver_action, make_break_guard_action,
    make_forced_pass_action, make_clear_action, make_set_guard_action,
    make_use_resource_action,
    make_squad_scout_action, make_squad_clear_action,
    make_squad_reinforce_action, make_squad_weaken_action,
    make_rush_protect_action, make_rush_speed_action,
)

logger = logging.getLogger("lychee_client.strategy")


def _make_process_action(
    match_id: str,
    round_num: int,
    player_id: int,
    process_type: str,
    current_node_id: str,
    phase: str,
) -> dict:
    """Map processType to the correct protocol action."""
    if process_type == "DOCK":
        return make_action(match_id, round_num, player_id, [make_dock_action(current_node_id)])
    if is_verify_process(process_type):
        if phase == "RUSH":
            return make_action(match_id, round_num, player_id, [make_verify_gate_action(current_node_id)])
        return make_empty_action(match_id, round_num, player_id)
    return make_action(match_id, round_num, player_id, [make_process_action(current_node_id)])


def _append_squad_action(
    action_msg: dict,
    squad_action: dict | None,
) -> dict:
    if squad_action is None:
        return action_msg
    actions = action_msg.get("msg_data", {}).get("actions", [])
    if len(actions) >= 2:
        return action_msg
    if len(actions) == 1:
        actions = actions + [squad_action]
    else:
        actions = [squad_action]
    action_msg["msg_data"]["actions"] = actions
    return action_msg


def decide_action(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node: dict | None = None,
    process_nodes: dict[str, dict] | None = None,
    contests: list[dict] | None = None,
    events: list[dict] | None = None,
    active_contest_id: str = "",
    last_move_failed: bool = False,
    last_move_error: str = "",
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    tasks: list[dict] | None = None,
    phase: str = "",
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    weather: dict | None = None,
    all_players: list[dict] | None = None,
    inquire_nodes: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
    pending_task_hold_task_id: str = "",
    pending_task_hold_node_id: str = "",
    pending_task_hold_until_round: int = 0,
    forced_pass_failed_targets: set[str] | None = None,
    failed_intel_targets: set[str] | None = None,
    scouted_node_ids: set[str] | None = None,
    bounties: list[dict] | None = None,
) -> dict:
    """Decide the action for the current round.

    Implements the single-frame decision pseudocode from 策略文档 §14.
    Returns a complete action message dict.
    """
    # Defaults
    if terminal_node_ids is None:
        terminal_node_ids = []
    if tasks is None:
        tasks = []
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()
    if weather is None:
        weather = {}
    if all_players is None:
        all_players = []
    if inquire_nodes is None:
        inquire_nodes = []
    if failed_task_ids is None:
        failed_task_ids = set()
    if guard_blocked_targets is None:
        guard_blocked_targets = set()
    if avoid_route_nodes is None:
        avoid_route_nodes = set()
    if forced_pass_failed_targets is None:
        forced_pass_failed_targets = set()
    if failed_intel_targets is None:
        failed_intel_targets = set()
    if scouted_node_ids is None:
        scouted_node_ids = set()
    if bounties is None:
        bounties = []

    try:
        deferred_scout: list[dict] = []
        result = _decide_action_impl(
            match_id, round_num, player_id, player, graph,
            current_node, process_nodes, contests, events,
            active_contest_id, last_move_failed, last_move_error,
            gate_node_id, terminal_node_ids, tasks, phase,
            processed_node_ids, visited_node_ids, weather, all_players, inquire_nodes,
            failed_task_ids, rush_speed_failed, guard_blocked_targets, avoid_route_nodes,
            pending_task_hold_task_id, pending_task_hold_node_id, pending_task_hold_until_round,
            forced_pass_failed_targets, failed_intel_targets, scouted_node_ids, bounties,
            deferred_scout,
        )
        if deferred_scout:
            return _append_squad_action(result, deferred_scout[0])
        return result
    except Exception as e:
        logger.error("Round %d: Strategy error: %s", round_num, e, exc_info=True)
        return make_empty_action(match_id, round_num, player_id)


def _decide_action_impl(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node: dict | None = None,
    process_nodes: dict[str, dict] | None = None,
    contests: list[dict] | None = None,
    events: list[dict] | None = None,
    active_contest_id: str = "",
    last_move_failed: bool = False,
    last_move_error: str = "",
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    tasks: list[dict] | None = None,
    phase: str = "",
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    weather: dict | None = None,
    all_players: list[dict] | None = None,
    inquire_nodes: list[dict] | None = None,
    failed_task_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
    guard_blocked_targets: set[str] | None = None,
    avoid_route_nodes: set[str] | None = None,
    pending_task_hold_task_id: str = "",
    pending_task_hold_node_id: str = "",
    pending_task_hold_until_round: int = 0,
    forced_pass_failed_targets: set[str] | None = None,
    failed_intel_targets: set[str] | None = None,
    scouted_node_ids: set[str] | None = None,
    bounties: list[dict] | None = None,
    deferred_scout: list | None = None,
) -> dict:
    if guard_blocked_targets is None:
        guard_blocked_targets = set()
    if avoid_route_nodes is None:
        avoid_route_nodes = set()
    if forced_pass_failed_targets is None:
        forced_pass_failed_targets = set()
    if failed_intel_targets is None:
        failed_intel_targets = set()
    if scouted_node_ids is None:
        scouted_node_ids = set()
    if bounties is None:
        bounties = []
    if deferred_scout is None:
        deferred_scout = []

    obstacle_nodes: set[str] = set()
    for node in inquire_nodes:
        if node_has_obstacle(node):
            obstacle_nodes.add(node.get("nodeId", ""))

    # --- 小分队探路（与主车队共用导航目标，不写死节点编号）---
    if phase != "RUSH" and not _should_force_delivery(round_num, phase, player):
        squad_scout = _pick_squad_scout_target(
            graph, get_current_node_id(player) or "", gate_node_id, terminal_node_ids,
            process_nodes, processed_node_ids, visited_node_ids, scouted_node_ids,
            weather, player, obstacle_nodes,
        )
        if squad_scout:
            scout_item = make_squad_scout_action(squad_scout)
            state = player.get("state", "")
            logger.info("Round %d: Squad scout at %s", round_num, squad_scout)
            if state == "IDLE" and can_act(player):
                deferred_scout.append(scout_item)
            elif state in ("MOVING", "WAITING", "IDLE"):
                return _append_squad_action(
                    make_empty_action(match_id, round_num, player_id),
                    scout_item,
                )

    # --- P0: Stability ---
    if is_retired(player) or is_delivered(player):
        return make_empty_action(match_id, round_num, player_id)

    state = player.get("state", "")
    current_node_id = get_current_node_id(player)
    my_team_id = get_team_id(player)

    # If in CONTESTING state, we must send WINDOW_CARD
    if state == "CONTESTING":
        on_water_route = _is_on_water_route(graph, current_node_id, gate_node_id, terminal_node_ids)
        return _handle_contesting(
            match_id, round_num, player_id, player,
            contests, events, active_contest_id, player,
            all_players, phase, on_water_route,
        )

    # Passive states: PROCESSING, VERIFYING, FORCED_PASSING, RESTING → heartbeat
    if is_in_passive_state(player):
        return make_empty_action(match_id, round_num, player_id)

    blocked = get_blocked_nodes(inquire_nodes, my_team_id, player_id)
    route_blocked = set(blocked)
    route_blocked.update(guard_blocked_targets)
    route_blocked.update(avoid_route_nodes)
    opp_player = _find_opponent(all_players, player_id)
    mode = classify_opponent_mode(player, opp_player, phase, gate_node_id)

    force_delivery = _should_force_delivery(round_num, phase, player)

    if is_in_limited_state(player):
        guard_target = _resolve_guard_block_target(player, route_blocked, guard_blocked_targets)

        if state == "WAITING":
            next_node = player.get("nextNodeId", "")
            if last_move_failed and last_move_error == "OBJECT_BUSY":
                logger.info("Round %d: OBJECT_BUSY in WAITING, sending WAIT", round_num)
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            pending_process_type = _get_pending_station_process_type(
                current_node_id, next_node, process_nodes, processed_node_ids,
            )
            if pending_process_type:
                if _has_current_process_for_node(player, current_node_id):
                    logger.info("Round %d: station process running at %s, sending empty action", round_num, current_node_id)
                    return make_empty_action(match_id, round_num, player_id)
                logger.info("Round %d: station process not started at %s, retrying %s", round_num, current_node_id, pending_process_type)
                return _make_process_action(
                    match_id, round_num, player_id,
                    pending_process_type, current_node_id, phase,
                )

            if last_move_failed and last_move_error == "PROCESS_REQUIRED":
                process_type = process_nodes.get(current_node_id, {}).get("processType") if process_nodes and current_node_id else ""
                if process_type:
                    logger.info("Round %d: PROCESS_REQUIRED in WAITING at %s, retrying %s", round_num, current_node_id, process_type)
                    return _make_process_action(
                        match_id, round_num, player_id,
                        process_type, current_node_id, phase,
                    )
                logger.info("Round %d: PROCESS_REQUIRED in WAITING at %s, sending WAIT", round_num, current_node_id)
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            if not force_delivery and current_node_id and not next_node:
                if (
                    pending_task_hold_node_id == current_node_id
                    and round_num <= pending_task_hold_until_round
                ):
                    logger.info(
                        "Round %d: waiting for busy task at %s until %d",
                        round_num, current_node_id, pending_task_hold_until_round,
                    )
                    return make_action(match_id, round_num, player_id, [make_wait_action()])
                if pending_task_hold_node_id == current_node_id and pending_task_hold_task_id:
                    task_retry = _retry_task_at_current_node(
                        match_id, round_num, player_id, player, graph,
                        current_node_id, tasks, failed_task_ids,
                        preferred_task_id=pending_task_hold_task_id,
                    )
                    if task_retry is not None:
                        return task_retry
                task_retry = _retry_task_at_current_node(
                    match_id, round_num, player_id, player, graph,
                    current_node_id, tasks, failed_task_ids,
                )
                if task_retry is not None:
                    return task_retry

            if force_delivery and current_node_id and not next_node:
                direct_target = _find_direct_delivery_step(
                    graph, current_node_id, player, gate_node_id, terminal_node_ids,
                    weather, process_nodes, processed_node_ids,
                )
                if direct_target:
                    if direct_target in route_blocked or direct_target in obstacle_nodes:
                        return _handle_force_delivery_blocker(
                            match_id, round_num, player_id, player,
                            direct_target, inquire_nodes, tasks, failed_task_ids,
                            obstacle_nodes, my_team_id,
                        )
                    choke_action = _handle_key_choke_forced_pass(
                        match_id, round_num, player_id,
                        current_node_id, direct_target, forced_pass_failed_targets,
                        inquire_nodes, graph, route_blocked, obstacle_nodes,
                        my_team_id, player_id,
                    )
                    if choke_action is not None:
                        return choke_action
                    horse_action = _handle_key_choke_horse(
                        match_id, round_num, player_id, player,
                        current_node_id, direct_target, inquire_nodes, graph,
                    )
                    if horse_action is not None:
                        return horse_action
                    logger.info("Round %d: FORCE_DELIVERY move to %s (WAITING)", round_num, direct_target)
                    return make_action(match_id, round_num, player_id, [make_move_action(direct_target)])

            if guard_target:
                return _wait_and_weaken_guard(
                    match_id, round_num, player_id, player,
                    inquire_nodes, guard_target, my_team_id,
                )

            if last_move_failed and last_move_error in ("OBJECT_BUSY", "MOVING_ACTION_FORBIDDEN"):
                logger.info("Round %d: %s in WAITING, sending WAIT", round_num, last_move_error)
                return make_action(match_id, round_num, player_id, [make_wait_action()])

            if next_node:
                if next_node in route_blocked:
                    return _wait_and_weaken_guard(
                        match_id, round_num, player_id, player,
                        inquire_nodes, next_node, my_team_id,
                    )
                return make_action(match_id, round_num, player_id, [make_move_action(next_node)])
            if current_node_id:
                move_target = _find_move_target(
                    graph, current_node_id, player, gate_node_id, terminal_node_ids,
                    weather, route_blocked, obstacle_nodes=obstacle_nodes,
                    process_nodes=process_nodes,
                    processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
                    round_num=round_num,
                )
                if move_target and move_target not in route_blocked:
                    return make_action(match_id, round_num, player_id, [make_move_action(move_target)])
                if move_target:
                    return _wait_and_weaken_guard(
                        match_id, round_num, player_id, player,
                        inquire_nodes, move_target, my_team_id,
                    )

        if state == "MOVING":
            if guard_target or last_move_failed and last_move_error == "MOVE_BLOCKED_BY_GUARD":
                target = guard_target or player.get("nextNodeId", "")
                return _wait_and_weaken_guard(
                    match_id, round_num, player_id, player,
                    inquire_nodes, target, my_team_id,
                )
            moving_action = _handle_moving(match_id, round_num, player_id, player, graph, weather, phase)
            if moving_action.get("msg_data", {}).get("actions"):
                return moving_action

        return make_empty_action(match_id, round_num, player_id)

    # Must be IDLE to act
    if not can_act(player):
        return make_empty_action(match_id, round_num, player_id)

    if current_node_id is None:
        return make_empty_action(match_id, round_num, player_id)

    if (
        not force_delivery
        and pending_task_hold_node_id == current_node_id
        and round_num <= pending_task_hold_until_round
    ):
        logger.info(
            "Round %d: holding at %s for busy task until %d",
            round_num, current_node_id, pending_task_hold_until_round,
        )
        return make_action(match_id, round_num, player_id, [make_wait_action()])
    if (
        not force_delivery
        and pending_task_hold_node_id == current_node_id
        and pending_task_hold_task_id
    ):
        task_retry = _retry_task_at_current_node(
            match_id, round_num, player_id, player, graph,
            current_node_id, tasks, failed_task_ids,
            preferred_task_id=pending_task_hold_task_id,
        )
        if task_retry is not None:
            return task_retry

    # Don't use blocked_nodes as hard filter in BFS — it causes TARGET_NOT_REACHABLE
    # Instead, use weighted routing to prefer unblocked paths
    blocked_soft = route_blocked  # used for weighted routing and combat

    # --- P1: Delivery flow (策略文档 §4.2 FSM) ---

    # At terminal: DELIVER if verified and can deliver
    if current_node_id in terminal_node_ids:
        if is_verified(player) and get_good_fruit(player) > 0 and get_freshness(player) > 0:
            return make_action(match_id, round_num, player_id, [make_deliver_action()])
        # 未验核则返回宫门（策略文档 §4.2: 无视设卡与障碍）
        if gate_node_id and not is_verified(player):
            step = graph.next_step_toward(current_node_id, gate_node_id, weather, None)
            if step:
                return make_action(match_id, round_num, player_id, [make_move_action(step)])
        return make_empty_action(match_id, round_num, player_id)

    # At gate: VERIFY_GATE in RUSH phase
    if gate_node_id and is_at_node(player, gate_node_id) and not is_verified(player):
        if phase == "RUSH":
            action = make_verify_gate_action(current_node_id)
            if get_bad_fruit(player) >= 2 or get_good_fruit(player) >= 1:
                action["rushTactic"] = "BREAK_ORDER"
                logger.info("Round %d: VERIFY_GATE with BREAK_ORDER at %s", round_num, current_node_id)
            return make_action(match_id, round_num, player_id, [action])
        # Not RUSH yet: don't submit VERIFY_GATE (will be rejected)
        # Continue doing other things until RUSH

    # --- Fixed processing (策略文档 §4.1: 再次到达同一站需重新处理) ---
    # Process at current node ONLY if not already processed this visit.
    # processed_node_ids tracks nodes where we completed processing this session.
    # If already processed, skip to MOVE (even if node has processType).
    already_processed_here = current_node_id in processed_node_ids
    process_type = None if already_processed_here else _get_process_type(current_node, process_nodes, current_node_id)

    if process_type:
        if last_move_failed and "WINDOW" in last_move_error.upper():
            move_target = _find_move_target(
                graph, current_node_id, player, gate_node_id, terminal_node_ids,
                weather, route_blocked, obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
                processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
                round_num=round_num,
            )
            if move_target:
                return make_action(match_id, round_num, player_id, [make_move_action(move_target)])
            return make_empty_action(match_id, round_num, player_id)

        return _make_process_action(match_id, round_num, player_id, process_type, current_node_id, phase)

    if last_move_failed and last_move_error == "PROCESS_REQUIRED":
        process_type = _get_process_type(current_node, process_nodes, current_node_id)
        if process_type:
            logger.info("Round %d: PROCESS_REQUIRED at %s, sending %s", round_num, current_node_id, process_type)
            return _make_process_action(match_id, round_num, player_id, process_type, current_node_id, phase)
        return make_action(match_id, round_num, player_id, [make_process_action(current_node_id)])

    # --- Handle OBJECT_BUSY: wait one round and retry ---
    if last_move_failed and last_move_error == "OBJECT_BUSY":
        # The process target is busy (e.g., window contest just ended, still transitioning)
        # Wait one round, then retry process on next round
        logger.info("Round %d: OBJECT_BUSY, waiting", round_num)
        return make_action(match_id, round_num, player_id, [make_wait_action()])

    # --- Handle blocked movement ---
    if last_move_failed and last_move_error == "MOVE_BLOCKED_BY_GUARD":
        return _handle_blocked_by_guard(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, blocked_soft, inquire_nodes, process_nodes=process_nodes,
        )

    # --- P2: Task strategy (策略文档 §5) ---
    if not force_delivery:
        task_action = _handle_tasks(
            match_id, round_num, player_id, player, graph,
            current_node_id, tasks, player_id, phase, weather, blocked,
            goal_node_id=gate_node_id, terminal_node_ids=terminal_node_ids,
            obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
            processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
            failed_task_ids=failed_task_ids,
        )
        if task_action is not None:
            return task_action

    # --- P4: Resource strategy (策略文档 §6) ---
    # Skip resource claiming when close to gate (prioritize delivery)
    dist_to_gate = 0
    if gate_node_id:
        dist_to_gate = graph.path_length(current_node_id, gate_node_id, weather, None)
    if not force_delivery and dist_to_gate > 4:  # Only claim resources when not close to gate
        resource_action = _handle_resources(
            match_id, round_num, player_id, player, graph,
            current_node_id, current_node, phase, weather,
        )
        if resource_action is not None:
            return resource_action
    if force_delivery:
        resource_action = _handle_force_delivery_resource(
            match_id, round_num, player_id, player, graph,
            current_node_id, current_node, gate_node_id,
            terminal_node_ids, weather, process_nodes, processed_node_ids,
            inquire_nodes,
        )
        if resource_action is not None:
            return resource_action

    # --- P5: Use resources (ice box, horses, intel) ---
    use_res_action = _handle_use_resources(
        match_id, round_num, player_id, player,
        current_node_id, graph, weather, phase, process_nodes,
        processed_node_ids, visited_node_ids, failed_intel_targets,
        last_move_failed, last_move_error,
    )
    if use_res_action is not None:
        return use_res_action

    # --- P5: Combat (策略文档 §8) — guard, break, squad ---
    if not force_delivery:
        combat_action = _handle_combat(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, blocked_soft, mode, phase, inquire_nodes, opp_player,
            obstacle_nodes=obstacle_nodes,
            process_nodes=process_nodes,
            visited_node_ids=visited_node_ids,
            my_team_id=my_team_id,
            bounties=bounties,
        )
        if combat_action is not None:
            return combat_action

    # --- Rush tactics (策略文档 §10) ---
    rush_action = _handle_rush_tactics(
        match_id, round_num, player_id, player,
        current_node_id, phase, mode,
        graph=graph, gate_node_id=gate_node_id,
        terminal_node_ids=terminal_node_ids, weather=weather,
        obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
        processed_node_ids=processed_node_ids, visited_node_ids=visited_node_ids,
        rush_speed_failed=rush_speed_failed,
    )
    if rush_action is not None:
        return rush_action

    # --- NAVIGATION: Move toward goal ---
    if force_delivery:
        direct_target = _find_direct_delivery_step(
            graph, current_node_id, player, gate_node_id, terminal_node_ids,
            weather, process_nodes, processed_node_ids,
        )
        if direct_target:
            if direct_target in route_blocked or direct_target in obstacle_nodes:
                blocker_action = _handle_force_delivery_blocker(
                    match_id, round_num, player_id, player,
                    direct_target, inquire_nodes, tasks, failed_task_ids,
                    obstacle_nodes, my_team_id,
                )
                if blocker_action.get("msg_data", {}).get("actions"):
                    return blocker_action
            choke_action = _handle_key_choke_forced_pass(
                match_id, round_num, player_id,
                current_node_id, direct_target, forced_pass_failed_targets,
                inquire_nodes, graph, route_blocked, obstacle_nodes,
                my_team_id, player_id,
            )
            if choke_action is not None:
                return choke_action
            horse_action = _handle_key_choke_horse(
                match_id, round_num, player_id, player,
                current_node_id, direct_target, inquire_nodes, graph,
            )
            if horse_action is not None:
                return horse_action
            logger.info("Round %d: FORCE_DELIVERY move to %s (goal=%s)", round_num, direct_target, gate_node_id)
            return make_action(match_id, round_num, player_id, [make_move_action(direct_target)])

    move_target = _find_move_target(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, route_blocked, obstacle_nodes=obstacle_nodes, process_nodes=process_nodes,
        processed_node_ids=processed_node_ids,
        visited_node_ids=set() if force_delivery else visited_node_ids,
        round_num=round_num,
    )

    # Next hop has enemy guard → break / forced pass / detour before MOVE
    if move_target and move_target in route_blocked:
        guard_action = _handle_blocked_by_guard(
            match_id, round_num, player_id, player, graph,
            current_node_id, gate_node_id, terminal_node_ids,
            weather, route_blocked, inquire_nodes, process_nodes=process_nodes,
        )
        if guard_action.get("msg_data", {}).get("actions"):
            return guard_action

    # Handle obstacle on next step (策略文档 §3.4: 道路障碍 → T04/CLEAR/FORCED_PASS)
    if move_target and move_target in obstacle_nodes:
        # Priority 1: CLAIM_TASK if T04 task exists at obstacle node (score + clear)
        t04_task = None
        for task in tasks:
            if (task.get("nodeId") == move_target
                    and task.get("active", False)
                    and not task.get("completed", False)
                    and not task.get("failed", False)
                    and get_task_template_id(task).startswith("T04")
                    and task.get("taskId", "") not in failed_task_ids):
                t04_task = task
                break
        if t04_task:
            logger.info("Round %d: Obstacle at %s, claiming T04 task", round_num, move_target)
            return make_action(match_id, round_num, player_id, [
                make_claim_task_action(t04_task.get("taskId", ""))
            ])

        # Priority 2: CLEAR if we have good fruit to spare (策略文档 §3.4: 1好果6帧)
        good_fruit = get_good_fruit(player)
        if good_fruit >= 2:  # Reserve at least 1 for DELIVER
            logger.info("Round %d: Obstacle at %s, using CLEAR", round_num, move_target)
            return make_action(match_id, round_num, player_id, [
                make_clear_action(move_target)
            ])

        # Priority 3: FORCED_PASS for obstacle-only (策略文档 §3.4: 8帧, no fruit cost)
        logger.info("Round %d: Obstacle at %s, using FORCED_PASS", round_num, move_target)
        return make_action(match_id, round_num, player_id, [
            make_forced_pass_action(move_target)
        ])
    if move_target:
        logger.info("Round %d: NAV move to %s (goal=%s)", round_num, move_target,
                     gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "?"))
        return make_action(match_id, round_num, player_id, [make_move_action(move_target)])

    return make_empty_action(match_id, round_num, player_id)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _find_opponent(all_players: list[dict], my_player_id: int) -> dict | None:
    """Find the opponent player dict."""
    for p in all_players:
        if p.get("playerId") != my_player_id:
            return p
    return None


def _is_on_water_route(
    graph: MapGraph, current_node_id: str,
    gate_node_id: str, terminal_node_ids: list[str],
) -> bool:
    """Check if the current path to goal goes through water edges."""
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if not goal or not current_node_id:
        return False
    path = graph.weighted_shortest_path(current_node_id, goal)
    if not path:
        return False
    for i in range(len(path) - 1):
        if graph.get_edge_route_type(path[i], path[i + 1]) == "WATER":
            return True
    return False


def _get_weather_penalized_routes(weather: dict) -> set[str]:
    """Get route types that should be penalized/avoided based on weather forecast.

    Returns set of route types to avoid (策略文档 §3.2, §6).
    """
    avoid = set()
    if not weather:
        return avoid
    forecasts = weather.get("forecast", []) + weather.get("active", [])
    for fw in forecasts:
        wtype = fw.get("type", "")
        region = fw.get("region", "")
        if wtype == "HOT" or region in ("ALL", "HOT"):
            avoid.add("MOUNTAIN")
        elif wtype == "HEAVY_RAIN" or region in ("WATER", "HEAVY_RAIN"):
            avoid.add("WATER")
        elif wtype == "MOUNTAIN_FOG" or region in ("MOUNTAIN", "MOUNTAIN_FOG"):
            avoid.add("MOUNTAIN")
    return avoid


def _remaining_fixed_process_nodes(
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
) -> dict[str, dict]:
    """未完成的固定处理站（排除验核站），供寻路与探路共用。"""
    if not process_nodes:
        return {}
    return {
        nid: info for nid, info in process_nodes.items()
        if nid not in processed_node_ids
        and not is_verify_process(info.get("processType"))
    }


def _nodes_with_process_type(
    process_nodes: dict[str, dict] | None,
    *process_types: str,
) -> set[str]:
    if not process_nodes:
        return set()
    allowed = set(process_types)
    return {
        nid for nid, info in process_nodes.items()
        if info.get("processType") in allowed
    }


def _estimate_rest_route_cost(
    graph: MapGraph,
    from_node: str,
    goal: str,
    weather: dict | None,
    blocked: set[str],
    remaining_process_nodes: dict[str, dict] | None,
    obstacle_penalty_nodes: set[str] | None = None,
) -> float | None:
    """估算从 from_node 到 goal 的后续路线耗时（宏观官道/水路语义）。

    - 下一跳是登船站(BOARD/DOCK)：后续必须经水路换运(WATER_TRANSFER)，不能走 S04→S07 陆路捷径
    - 官道分支：寻路时绕开未完成的登船站，避免误把陆路捷径算成「水路」
    """
    if not remaining_process_nodes:
        remaining_process_nodes = {}
    if obstacle_penalty_nodes is None:
        obstacle_penalty_nodes = set()

    def _edge_cost(from_id: str, to_id: str) -> float:
        cost = graph.edge_cost(from_id, to_id, weather, blocked, remaining_process_nodes)
        if to_id in obstacle_penalty_nodes:
            cost += 8  # 路障处理站：强制通行时间税
        return cost

    def _sum_path(path: list[str]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            total += _edge_cost(path[i], path[i + 1])
        return total

    proc = remaining_process_nodes.get(from_node, {}).get("processType", "")
    board_nodes = _nodes_with_process_type(remaining_process_nodes, "BOARD", "DOCK")
    water_transfer_nodes = _nodes_with_process_type(remaining_process_nodes, "WATER_TRANSFER")

    if proc in ("BOARD", "DOCK") and water_transfer_nodes:
        best_cost = float("inf")
        for wt in water_transfer_nodes:
            # 登船后优先走水路边到换运站（禁止把 S04→S07 陆路捷径算入水路）
            leg1 = graph.weighted_shortest_path(
                from_node, wt, weather, blocked, remaining_process_nodes,
            )
            if not leg1 and from_node in graph.get_neighbors(wt):
                leg1 = [from_node, wt]
            if not leg1:
                continue
            leg2 = graph.weighted_shortest_path(
                wt, goal, weather, blocked, remaining_process_nodes,
            )
            if not leg2:
                continue
            cost = _sum_path(leg1) + _sum_path(leg2)
            if cost < best_cost:
                best_cost = cost
        if best_cost < float("inf"):
            return best_cost

    road_blocked = set(blocked)
    if from_node not in board_nodes:
        road_blocked.update(board_nodes)
    rest = graph.weighted_shortest_path(
        from_node, goal, weather, road_blocked, remaining_process_nodes,
    )
    if rest:
        return _sum_path(rest)

    rest = graph.weighted_shortest_path(
        from_node, goal, weather, blocked, remaining_process_nodes,
    )
    if not rest:
        return None
    return _sum_path(rest)


def _get_process_type(
    current_node: dict | None,
    process_nodes: dict[str, dict] | None,
    current_node_id: str,
) -> str | None:
    """Get the process type for the current node."""
    if current_node and needs_processing(current_node):
        return current_node.get("processType", "")
    if process_nodes and current_node_id in process_nodes:
        pn = process_nodes[current_node_id]
        return pn.get("processType", "")
    return None


def _get_goal_node(
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    graph: MapGraph,
    current_node_id: str,
    weather: dict | None = None,
    blocked: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
) -> str | None:
    """Determine the current navigation goal based on player state."""
    if is_delivered(player):
        return None
    if not is_verified(player) and gate_node_id:
        return gate_node_id
    if is_verified(player) and terminal_node_ids:
        # Find nearest terminal via weighted path
        best = None
        best_cost = float('inf')
        for tid in terminal_node_ids:
            path = graph.weighted_shortest_path(current_node_id, tid, weather, blocked, process_nodes)
            if path:
                cost = sum(graph.edge_cost(path[i], path[i+1], weather, blocked, process_nodes)
                           for i in range(len(path)-1))
                if cost < best_cost:
                    best_cost = cost
                    best = tid
        return best
    return None


def _should_force_delivery(round_num: int, phase: str, player: dict) -> bool:
    """Stop optional scoring once delivery risk is higher than task/resource value."""
    if phase == "RUSH":
        return True
    task_score = get_task_score(player)
    if round_num >= 520:
        return True
    if round_num >= 400 and task_score >= TASK_SCORE_TARGET:
        return True
    if round_num >= 350 and task_score >= TASK_SCORE_STRETCH:
        return True
    return False


def _is_approaching_key_pass(
    current_node_id: str,
    target_node_id: str,
    inquire_nodes: list[dict],
    graph: MapGraph,
) -> bool:
    """下一跳是否为关键关隘（不写死节点编号）。"""
    target_node = find_node_by_id(inquire_nodes, target_node_id)
    if not is_key_pass_node(target_node):
        return False
    return target_node_id in graph.get_neighbors(current_node_id)


def _edge_distance(graph: MapGraph, from_id: str, to_id: str) -> int:
    edge = graph.get_edge(from_id, to_id)
    if edge:
        return int(edge.get("distance", 30))
    return 30


def _node_coord_distance(graph: MapGraph, from_id: str, to_id: str) -> int:
    """沿路线边累计距离（任务书 §3.3.4，不用坐标直线距离）。"""
    return graph.route_distance(from_id, to_id)


def _target_needs_forced_pass(
    target_node_id: str,
    route_blocked: set[str],
    obstacle_nodes: set[str],
    inquire_nodes: list[dict],
    my_team_id: str,
    player_id: int,
) -> bool:
    """仅当相邻目标存在设卡或障碍时才需要强制通行。"""
    if target_node_id in obstacle_nodes or target_node_id in route_blocked:
        return True
    for node in inquire_nodes:
        if node.get("nodeId") != target_node_id:
            continue
        if node_has_obstacle(node):
            return True
        if is_enemy_guard(node.get("guard"), my_team_id, player_id):
            return True
    return False


def _find_bounty_node(bounties: list[dict], my_team_id: str) -> str:
    """返回敌方设卡且有悬赏的节点（可优先攻坚）。"""
    for bounty in bounties:
        if bounty.get("winnerPlayerId", 0):
            continue
        if bounty.get("ownerTeamId") == my_team_id:
            continue
        node_id = bounty.get("nodeId", "")
        if node_id:
            return node_id
    return ""


def _has_active_speed_buff(player: dict) -> bool:
    for buff in player.get("buffs", []) or []:
        if buff.get("type") in ("FAST_HORSE", "SHORT_HORSE", "RUSH_SPEED"):
            return True
    return False


def _find_direct_delivery_step(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
) -> str | None:
    goal_node = _get_goal_node(
        player, gate_node_id, terminal_node_ids, graph,
        current_node_id, weather, None, process_nodes,
    )
    if not goal_node:
        return None

    remaining_process_nodes = None
    if process_nodes:
        remaining_process_nodes = {
            nid: info for nid, info in process_nodes.items()
            if nid not in processed_node_ids
        }

    # Ignore guards/obstacles here. If the direct next hop is blocked, handle
    # that blocker explicitly instead of oscillating through detours.
    step = graph.next_step_toward(
        current_node_id, goal_node, weather, None,
        use_weighted=True, process_nodes=remaining_process_nodes,
    )
    if step:
        return step
    return graph.next_step_toward(current_node_id, goal_node, weather, None, use_weighted=False)


def _handle_force_delivery_blocker(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    target_node_id: str,
    inquire_nodes: list[dict],
    tasks: list[dict],
    failed_task_ids: set[str],
    obstacle_nodes: set[str],
    my_team_id: str,
) -> dict:
    if target_node_id in obstacle_nodes:
        t04_task = None
        for task in tasks:
            if (task.get("nodeId") == target_node_id
                    and task.get("active", False)
                    and not task.get("completed", False)
                    and not task.get("failed", False)
                    and get_task_template_id(task).startswith("T04")
                    and task.get("taskId", "") not in failed_task_ids):
                t04_task = task
                break
        if t04_task:
            logger.info("Round %d: FORCE_DELIVERY T04 clear at %s", round_num, target_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_task_action(t04_task.get("taskId", ""))
            ])
        if get_good_fruit(player) >= 2:
            logger.info("Round %d: FORCE_DELIVERY CLEAR at %s", round_num, target_node_id)
            return make_action(match_id, round_num, player_id, [make_clear_action(target_node_id)])
        logger.info("Round %d: FORCE_DELIVERY forced pass obstacle at %s", round_num, target_node_id)
        return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])

    for node in inquire_nodes:
        if node.get("nodeId") != target_node_id:
            continue
        guard = node.get("guard", {})
        if is_enemy_guard(guard, my_team_id, player_id):
            good = min(get_good_fruit(player), 2)
            bad = min(get_bad_fruit(player), 2)
            if good + bad > 0:
                action = make_break_guard_action(target_node_id, good_fruit=good, bad_fruit=bad)
                logger.info("Round %d: FORCE_DELIVERY break guard at %s", round_num, target_node_id)
                return make_action(match_id, round_num, player_id, [action])
            logger.info("Round %d: FORCE_DELIVERY forced pass guard at %s", round_num, target_node_id)
            return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])

    logger.info("Round %d: FORCE_DELIVERY forced pass blocked %s", round_num, target_node_id)
    return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])


def _handle_key_choke_forced_pass(
    match_id: str,
    round_num: int,
    player_id: int,
    current_node_id: str,
    target_node_id: str,
    forced_pass_failed_targets: set[str],
    inquire_nodes: list[dict],
    graph: MapGraph | None = None,
    route_blocked: set[str] | None = None,
    obstacle_nodes: set[str] | None = None,
    my_team_id: str = "",
    my_player_id: int = 0,
) -> dict | None:
    """关键关隘前：仅在有阻挡时尝试强制通行，失败则改普通移动。"""
    if graph is None:
        return None
    if route_blocked is None:
        route_blocked = set()
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if not _is_approaching_key_pass(current_node_id, target_node_id, inquire_nodes, graph):
        return None
    if target_node_id in forced_pass_failed_targets:
        return None
    if not _target_needs_forced_pass(
        target_node_id, route_blocked, obstacle_nodes,
        inquire_nodes, my_team_id, my_player_id,
    ):
        return None
    logger.info("Round %d: forced pass probe at key pass %s", round_num, target_node_id)
    return make_action(match_id, round_num, player_id, [make_forced_pass_action(target_node_id)])


def _handle_key_choke_horse(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    current_node_id: str,
    target_node_id: str,
    inquire_nodes: list[dict],
    graph: MapGraph,
) -> dict | None:
    """进入关键关隘前使用快马/短程马。"""
    if not _is_approaching_key_pass(current_node_id, target_node_id, inquire_nodes, graph):
        return None
    edge_dist = _edge_distance(graph, current_node_id, target_node_id)
    if edge_dist < 30:
        return None
    if has_resource(player, "FAST_HORSE"):
        logger.info("Round %d: using FAST_HORSE before key pass %s", round_num, target_node_id)
        return make_action(match_id, round_num, player_id, [make_use_resource_action("FAST_HORSE")])
    if has_resource(player, "SHORT_HORSE"):
        logger.info("Round %d: using SHORT_HORSE before key pass %s", round_num, target_node_id)
        return make_action(match_id, round_num, player_id, [make_use_resource_action("SHORT_HORSE")])
    return None


def _pick_cheapest_first_hop(
    graph: MapGraph,
    current_node_id: str,
    goal_node: str,
    available: list[str],
    weather: dict | None,
    path_blocked: set[str],
    remaining_process_nodes: dict[str, dict] | None,
    avoid_routes: set[str] | None = None,
    log_compare: bool = False,
    obstacle_penalty_nodes: set[str] | None = None,
) -> str | None:
    """Compare each available neighbor by total weighted cost to goal.

    path_blocked 仅含道路障碍节点；visited/设卡不参与代价估算（否则会算不出路径）。
    """
    best_neighbor = None
    best_cost = float("inf")
    comparisons: list[tuple[str, float, str]] = []
    if obstacle_penalty_nodes is None:
        obstacle_penalty_nodes = set()
    board_nodes = _nodes_with_process_type(remaining_process_nodes or {}, "BOARD", "DOCK")

    for n in available:
        route_type = graph.get_edge_route_type(current_node_id, n)
        hop_cost = graph.edge_cost(
            current_node_id, n, weather, None, remaining_process_nodes,
        )
        if n in obstacle_penalty_nodes:
            hop_cost += 8
        if hop_cost == float("inf"):
            continue
        if avoid_routes and route_type in avoid_routes:
            hop_cost *= 1.35
        rest_cost = _estimate_rest_route_cost(
            graph, n, goal_node, weather, path_blocked, remaining_process_nodes,
            obstacle_penalty_nodes,
        )
        if rest_cost is None:
            continue
        total = hop_cost + rest_cost
        comparisons.append((n, total, route_type))
        if total < best_cost:
            best_cost = total
            best_neighbor = n

    if best_neighbor and len(comparisons) >= 2:
        comparisons.sort(key=lambda x: x[1])
        top_cost = comparisons[0][1]
        # 耗时接近时优先走登船/水路入口（差 8 帧以内）
        for nid, cost, _rt in comparisons:
            if cost - top_cost <= 8 and nid in board_nodes:
                best_neighbor = nid
                break

    if log_compare and len(available) > 1:
        comparisons.sort(key=lambda x: x[1])
        logger.info(
            "route compare at %s -> %s: %s (pick %s)",
            current_node_id,
            goal_node,
            ", ".join(f"{nid}={cost:.1f}({rt})" for nid, cost, rt in comparisons) if comparisons else "none",
            best_neighbor,
        )
    return best_neighbor


def _find_move_target(
    graph: MapGraph,
    current_node_id: str,
    player: dict,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None = None,
    blocked: set[str] | None = None,
    failed_target: str = "",
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    round_num: int = 0,
) -> str | None:
    """Find the best move target using weighted shortest path toward the current goal.

    Filters out obstacle nodes (hasObstacle=true) from move targets,
    since MOVE to an obstacle node will be rejected with TARGET_NOT_REACHABLE.
    """
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()

    neighbors = graph.get_neighbors(current_node_id)
    if not neighbors:
        return None

    # Filter out failed target, obstacle nodes, and enemy-guarded nodes when alternatives exist
    guard_blocked = blocked or set()
    available = [n for n in neighbors if n != failed_target and n not in obstacle_nodes]
    if guard_blocked:
        safe = [n for n in available if n not in guard_blocked]
        if safe:
            available = safe
    forward_available = [n for n in available if n not in visited_node_ids]
    if forward_available:
        available = forward_available
    logger.info("_find_move_target: current=%s neighbors=%s available=%s visited=%s failed_target=%s",
                current_node_id, neighbors, available, visited_node_ids, failed_target)
    if not available:
        # Fall back: allow backtrack but still avoid guarded nodes if possible
        available = [n for n in neighbors if n != failed_target and n not in obstacle_nodes]
        if guard_blocked:
            safe = [n for n in available if n not in guard_blocked]
            if safe:
                available = safe
        if not available:
            available = neighbors

    remaining_process_nodes = _remaining_fixed_process_nodes(process_nodes, processed_node_ids)

    goal_node = _get_goal_node(
        player, gate_node_id, terminal_node_ids, graph,
        current_node_id, weather, None, remaining_process_nodes,
    )

    if goal_node:
        # 路线比价：路障节点加 8 帧强过惩罚，但不阻断寻路（否则 rest=None 回退陆路）
        path_blocked: set[str] = set()
        obstacle_penalty_nodes = set(obstacle_nodes)
        soft_blocked = set(obstacle_nodes)
        soft_blocked.update(visited_node_ids)
        soft_blocked.update(guard_blocked)
        soft_blocked.discard(goal_node)

        avoid_routes = _get_weather_penalized_routes(weather or {})
        log_compare = len(available) > 1
        step = _pick_cheapest_first_hop(
            graph, current_node_id, goal_node, available, weather,
            path_blocked, remaining_process_nodes, avoid_routes,
            log_compare=log_compare,
            obstacle_penalty_nodes=obstacle_penalty_nodes,
        )
        if step:
            return step

        # Fallback: Dijkstra 首跳（实际移动仍不可进纯路障节点，由 available 过滤）
        step = graph.next_step_toward(
            current_node_id, goal_node, weather, obstacle_nodes,
            use_weighted=True, process_nodes=remaining_process_nodes,
        )
        if step and step in available:
            return step
        step = graph.next_step_toward(
            current_node_id, goal_node, weather, obstacle_nodes,
            use_weighted=False, process_nodes=remaining_process_nodes,
        )
        if step and step in available:
            return step
        step = graph.next_step_toward(
            current_node_id, goal_node, weather, None,
            use_weighted=True, process_nodes=remaining_process_nodes,
        )
        if step and step in available:
            return step
        step = graph.next_step_toward(
            current_node_id, goal_node, weather, None,
            use_weighted=False, process_nodes=remaining_process_nodes,
        )
        if step and step in available:
            return step

    # No goal: fall back to first available neighbor
    return available[0]


def _handle_contesting(
    match_id: str, round_num: int, player_id: int,
    player: dict, contests: list[dict] | None,
    events: list[dict] | None, active_contest_id: str,
    my_player: dict, all_players: list[dict], phase: str,
    on_water_route: bool = False,
) -> dict:
    """Handle CONTESTING state: choose window card (策略文档 §7)."""
    contest_id = _find_contest_id(player_id, contests, events, active_contest_id)
    if not contest_id:
        return make_empty_action(match_id, round_num, player_id)

    # Determine contest type and pick card
    contest = _find_contest(contest_id, contests)
    contest_type = ""
    if contest:
        contest_type = contest.get("contestType") or contest.get("type", "")

    card = _choose_window_card(
        contest_type, contest, my_player, all_players, phase, on_water_route,
    )
    return make_action(match_id, round_num, player_id, [
        make_window_card_action(contest_id, card)
    ])


def _find_contest_id(
    player_id: int,
    contests: list[dict] | None,
    events: list[dict] | None,
    active_contest_id: str,
) -> str:
    """Find the contest ID for the current player."""
    if active_contest_id:
        return active_contest_id
    if contests:
        for c in contests:
            if c.get("redPlayerId") == player_id or c.get("bluePlayerId") == player_id:
                if not c.get("resolved", False) and c.get("status") != "SUPPRESSED":
                    return c.get("contestId", "")
    if events:
        for ev in reversed(events):
            if ev.get("type") == "WINDOW_CONTEST_START":
                payload = ev.get("payload", {})
                cid = payload.get("contestId", "")
                if cid:
                    return cid
    return ""


def _find_contest(contest_id: str, contests: list[dict] | None) -> dict | None:
    """Find contest dict by ID."""
    if contests:
        for c in contests:
            if c.get("contestId") == contest_id:
                return c
    return None


def _choose_window_card(
    contest_type: str, contest: dict | None,
    my_player: dict, all_players: list[dict], phase: str,
    on_water_route: bool = False,
) -> str:
    """Choose window card based on contest type (策略文档 §7.3).

    克制关系: 验牒(YAN_DIE) 克 强行(QIANG_XING) 克 献贡(XIAN_GONG) 克 兵争(BING_ZHENG) 克 验牒
    """
    resources = get_player_resources(my_player)
    action_points = get_action_points(my_player)

    # Card availability
    has_yan_die = resources.get("PASS_TOKEN", 0) + resources.get("OFFICIAL_PERMIT", 0) > 0
    has_bing_zheng = action_points > 0
    has_xian_gong = get_good_fruit(my_player) >= 1 and get_freshness(my_player) >= 80
    has_qiang_xing = has_resource(my_player, "FAST_HORSE") or has_resource(my_player, "SHORT_HORSE")

    my_player_id = my_player.get("playerId", 0)
    opp = _find_opponent(all_players, my_player_id) if my_player_id else None
    mode = classify_opponent_mode(my_player, opp, phase)
    if mode == "CONSERVATIVE" and contest_type in ("RESOURCE", "DOCK"):
        return "ABSTAIN"

    # Strategy by contest type
    if contest_type == "GATE":
        if is_verified(my_player):
            return "ABSTAIN"
        # Must contest gate (策略文档 §7.3: GATE必争)
        if has_bing_zheng:
            return "BING_ZHENG"
        if has_yan_die:
            return "YAN_DIE"
        if has_xian_gong:
            return "XIAN_GONG"
        return "QIANG_XING"

    if contest_type == "RESOURCE":
        # Contest for important resources (fast horse, official permit)
        if has_yan_die:
            return "YAN_DIE"
        if has_bing_zheng:
            return "BING_ZHENG"
        return "ABSTAIN"  # Not critical enough to spend fruit

    if contest_type == "TASK":
        # Contest for 30-point tasks (策略文档 §7.3: 30分争, 15分不顺路弃权)
        if contest and contest.get("taskScore", 0) >= 30:
            if has_bing_zheng:
                return "BING_ZHENG"
            if has_yan_die:
                return "YAN_DIE"
            return "ABSTAIN"
        return "ABSTAIN"  # 15-point tasks not worth contesting

    if contest_type == "DOCK":
        # Only contest DOCK if on water route (策略文档 §7.3: 走水路必争, 官道路线弃权)
        if not on_water_route:
            return "ABSTAIN"
        if has_yan_die:
            return "YAN_DIE"
        if has_bing_zheng:
            return "BING_ZHENG"
        return "ABSTAIN"

    if contest_type == "PASS":
        # PASS contest: offensive (强行/验牒) vs defensive (策略文档 §7.3)
        # Attacker wants to pass → qiang_xing or yan_die
        # Defender → based on cost (abstain if time tax acceptable)
        if has_yan_die:
            return "YAN_DIE"
        if has_qiang_xing:
            return "QIANG_XING"
        return "ABSTAIN"

    if contest_type == "OBSTACLE":
        # Only contest for T04 (30 points) at obstacle
        if contest and contest.get("taskScore", 0) >= 30:
            if has_bing_zheng:
                return "BING_ZHENG"
            if has_yan_die:
                return "YAN_DIE"
        return "ABSTAIN"

    # Default: abstain
    return "ABSTAIN"


def _get_pending_station_process_type(
    current_node_id: str | None,
    next_node_id: str,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
) -> str:
    if not current_node_id or next_node_id or not process_nodes:
        return ""
    if current_node_id in processed_node_ids:
        return ""

    process_type = process_nodes.get(current_node_id, {}).get("processType")
    if process_type and not is_verify_process(process_type):
        return process_type
    return ""


def _has_current_process_for_node(player: dict, current_node_id: str | None) -> bool:
    if not current_node_id:
        return False
    current_process = player.get("currentProcess")
    if not isinstance(current_process, dict):
        return False
    target_node_id = current_process.get("targetNodeId", "")
    object_key = current_process.get("objectKey", "")
    return target_node_id == current_node_id or object_key.startswith(f"PROCESS:{current_node_id}:")


def _resolve_guard_block_target(
    player: dict,
    route_blocked: set[str],
    guard_blocked_targets: set[str],
) -> str:
    """Node blocking our in-progress move (next hop or known guard)."""
    next_node = player.get("nextNodeId", "")
    if next_node and next_node in route_blocked:
        return next_node
    return ""


def _make_squad_weaken_action(
    inquire_nodes: list[dict],
    target_node_id: str,
    my_team_id: str,
    player_id: int,
    player: dict,
) -> dict | None:
    if not target_node_id or get_squad_count(player) < 2:
        return None
    for node in inquire_nodes:
        if node.get("nodeId") != target_node_id:
            continue
        guard = node.get("guard", {})
        if is_enemy_guard(guard, my_team_id, player_id):
            if guard.get("defense", 0) > 0:
                return make_squad_weaken_action(target_node_id)
            return None
    # inquire 可能未包含远程节点，仍尝试削弱
    return make_squad_weaken_action(target_node_id)


def _wait_and_weaken_guard(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    inquire_nodes: list[dict],
    target_node_id: str,
    my_team_id: str,
) -> dict:
    """WAIT (主车队) + SQUAD_WEAKEN (小分队) 每帧削弱设卡直到通行。"""
    msg = make_action(match_id, round_num, player_id, [make_wait_action()])
    squad = _make_squad_weaken_action(
        inquire_nodes, target_node_id, my_team_id, player_id, player,
    )
    if squad:
        logger.info("Round %d: WAIT + squad weaken at %s", round_num, target_node_id)
        return _append_squad_action(msg, squad)
    return msg


def _handle_moving(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, weather: dict | None, phase: str,
) -> dict:
    """Handle MOVING state: use horse on long segments only."""
    if _has_active_speed_buff(player):
        return make_empty_action(match_id, round_num, player_id)

    current_node_id = get_current_node_id(player) or ""
    next_node = player.get("nextNodeId", "")
    if not current_node_id or not next_node:
        return make_empty_action(match_id, round_num, player_id)

    edge_dist = _edge_distance(graph, current_node_id, next_node)
    route_type = graph.get_edge_route_type(current_node_id, next_node)
    use_horse = edge_dist >= 35 or route_type in ("ROAD", "WATER")

    if use_horse and has_resource(player, "FAST_HORSE"):
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("FAST_HORSE")
        ])
    if use_horse and has_resource(player, "SHORT_HORSE"):
        return make_action(match_id, round_num, player_id, [
            make_use_resource_action("SHORT_HORSE")
        ])
    return make_empty_action(match_id, round_num, player_id)


def _handle_blocked_by_guard(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph,
    current_node_id: str, gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None, inquire_nodes: list[dict],
    process_nodes: dict[str, dict] | None = None,
) -> dict:
    """Handle MOVE_BLOCKED_BY_GUARD error (策略文档 §3.4)."""
    if blocked is None:
        blocked = set()
    neighbors = graph.get_neighbors(current_node_id)
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")

    # Detour via unblocked neighbor when direct hop is guarded
    best_detour = None
    best_cost = float("inf")
    for n in neighbors:
        if n in blocked:
            continue
        if not goal:
            return make_action(match_id, round_num, player_id, [make_move_action(n)])
        path = graph.weighted_shortest_path(n, goal, weather, blocked, process_nodes)
        if path:
            cost = sum(
                graph.edge_cost(path[i], path[i + 1], weather, blocked, process_nodes)
                for i in range(len(path) - 1)
            )
            if cost < best_cost:
                best_cost = cost
                best_detour = n
    if best_detour:
        logger.info("Round %d: Detour via %s to avoid guard", round_num, best_detour)
        return make_action(match_id, round_num, player_id, [make_move_action(best_detour)])

    # No detour: BREAK_GUARD or FORCED_PASS on guarded neighbor
    for n in neighbors:
        if n not in blocked:
            continue
        for node in inquire_nodes:
            if node.get("nodeId") != n:
                continue
            guard = node.get("guard", {})
            if is_enemy_guard(guard, get_team_id(player), player_id):
                good = min(get_good_fruit(player), 2)
                bad = min(get_bad_fruit(player), 2)
                if good + bad > 0:
                    logger.info(
                        "Round %d: BREAK_GUARD at %s (gf=%d bf=%d)",
                        round_num, n, good, bad,
                    )
                    return make_action(match_id, round_num, player_id, [
                        make_break_guard_action(n, good_fruit=good, bad_fruit=bad)
                    ])
        logger.info("Round %d: FORCED_PASS at guarded %s", round_num, n)
        return make_action(match_id, round_num, player_id, [
            make_forced_pass_action(n)
        ])

    return make_empty_action(match_id, round_num, player_id)


def _retry_task_at_current_node(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    tasks: list[dict],
    failed_task_ids: set[str],
    preferred_task_id: str = "",
) -> dict | None:
    if get_task_score(player) >= TASK_SCORE_TARGET:
        return None
    if isinstance(player.get("currentProcess"), dict):
        return None

    neighbors = graph.get_neighbors(current_node_id) if graph else None
    task = None
    if preferred_task_id:
        for candidate in tasks:
            if candidate.get("taskId", "") == preferred_task_id:
                task_node = candidate.get("nodeId", "")
                if (
                    task_node == current_node_id
                    or (neighbors is not None and task_node in neighbors and get_task_template_id(candidate).startswith("T04"))
                ):
                    task = candidate
                break
    if not task:
        task = find_task_at_node(
            tasks, current_node_id, player_id,
            graph_neighbors=neighbors,
        )
    if not task:
        return None
    if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
        return None
    owner = task.get("ownerPlayerId", 0)
    if owner != 0 and owner != player_id:
        return None
    protection = task.get("protectionPlayerId", 0)
    if protection != 0 and protection != player_id:
        return None

    task_id = task.get("taskId", "")
    if not task_id or task_id in failed_task_ids:
        return None
    template_id = get_task_template_id(task)
    if template_id.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
        return None
    expire_round = task.get("expireRound", 0)
    if expire_round > 0 and round_num >= expire_round:
        return None

    logger.info("Round %d: Retrying task %s (template=%s) at %s", round_num, task_id, template_id, current_node_id)
    return make_action(match_id, round_num, player_id, [make_claim_task_action(task_id)])


def _handle_tasks(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, current_node_id: str,
    tasks: list[dict], my_player_id: int, phase: str,
    weather: dict | None, blocked: set[str] | None,
    goal_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    failed_task_ids: set[str] | None = None,
) -> dict | None:
    """Handle task claiming strategy (策略文档 §5).

    Returns action dict or None.
    """
    if terminal_node_ids is None:
        terminal_node_ids = []
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()
    if failed_task_ids is None:
        failed_task_ids = set()

    my_task_score = get_task_score(player)
    if _should_force_delivery(round_num, phase, player):
        return None

    # Already at stretch target, don't need more tasks
    if my_task_score >= TASK_SCORE_STRETCH and phase != "RUSH":
        return None

    # Check if we're currently processing a task (策略文档 §5.2: 同时仅处理1个任务实例)
    for task in tasks:
        if (task.get("ownerPlayerId") == my_player_id
                and task.get("active", False)
                and not task.get("completed", False)
                and not task.get("failed", False)):
            return None

    # Try to claim task at current node (prioritized by score/round)
    task = find_task_at_node(
        tasks, current_node_id, my_player_id,
        graph_neighbors=graph.get_neighbors(current_node_id) if graph else None,
    )
    if task:
        template_id = get_task_template_id(task)
        if template_id.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
            logger.debug("Round %d: Skipping T06 task (no horse)", round_num)
            task = None

    if task:
        # Check expireRound (策略文档 §5.2: 关注expireRound)
        expire_round = task.get("expireRound", 0)
        if expire_round > 0 and round_num >= expire_round:
            logger.debug("Round %d: Task %s expired", round_num, task.get("taskId", ""))
            task = None

    if task:
        # Skip tasks previously rejected with RESOURCE_NOT_ENOUGH
        task_id = task.get("taskId", "")
        if task_id and task_id in failed_task_ids:
            logger.debug("Round %d: Skipping failed task %s", round_num, task_id)
            task = None

    if task:
        task_id = task.get("taskId", "")
        if task_id:
            logger.info("Round %d: Claiming task %s (template=%s) at %s", round_num, task_id, template_id, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_task_action(task_id)
            ])

    # Look for nearby tasks within detour cost (策略文档 §5.2 顺路原则)
    if my_task_score < TASK_SCORE_TARGET:
        candidates = []
        for task in tasks:
            if not task.get("active", False) or task.get("completed", False) or task.get("failed", False):
                continue
            owner = task.get("ownerPlayerId", 0)
            if owner != 0 and owner != my_player_id:
                continue
            protection = task.get("protectionPlayerId", 0)
            if protection != 0 and protection != my_player_id:
                continue
            task_node = task.get("nodeId", "")
            if not task_node:
                continue

            # T06: skip if no horse
            tid = get_task_template_id(task)
            if tid.startswith("T06") and not has_resource(player, "FAST_HORSE") and not has_resource(player, "SHORT_HORSE"):
                continue

            # Skip tasks previously rejected with RESOURCE_NOT_ENOUGH
            if task.get("taskId", "") in failed_task_ids:
                continue

            # Check expireRound
            expire_round = task.get("expireRound", 0)
            if expire_round > 0 and round_num >= expire_round:
                continue

            # Check detour cost
            detour = _calc_detour_cost(graph, current_node_id, task_node, goal_node_id, terminal_node_ids, weather, blocked, player, process_nodes)
            if detour <= MAX_TASK_DETOUR_COST:
                # Score per round priority (策略文档 §5.1)
                spr = 0.0
                for prefix, (score, proc_round, score_per_round) in TASK_PRIORITY.items():
                    if tid.startswith(prefix):
                        spr = score_per_round
                        break
                candidates.append((task, detour, spr))

        if candidates:
            # Sort by: score-per-round descending, then detour ascending
            candidates.sort(key=lambda x: (-x[2], x[1]))
            best_task = candidates[0][0]
            task_node = best_task.get("nodeId", "")
            # Move toward the task node using weighted routing, avoid backtracking
            soft_blocked = set(obstacle_nodes)
            soft_blocked.update(visited_node_ids)
            soft_blocked.discard(task_node)  # Don't block the target
            step = graph.next_step_toward(current_node_id, task_node, weather, soft_blocked, use_weighted=True, process_nodes=process_nodes)
            if not step:
                # Fallback without soft-blocked
                step = graph.next_step_toward(current_node_id, task_node, weather, obstacle_nodes, use_weighted=True, process_nodes=process_nodes)
            if step:
                logger.info("Round %d: Moving toward task at %s (template=%s), step=%s", round_num, task_node, get_task_template_id(best_task), step)
                return make_action(match_id, round_num, player_id, [make_move_action(step)])

    return None


def _calc_detour_cost(
    graph: MapGraph, current: str, task_node: str,
    gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None,
    player: dict,
    process_nodes: dict[str, dict] | None = None,
) -> int:
    """Calculate the extra weighted cost of detouring to a task node vs direct route."""
    goal = _get_goal_node(player, gate_node_id, terminal_node_ids, graph, current, weather, blocked, process_nodes)
    if not goal:
        return 999

    def _weighted_path_cost(a: str, b: str) -> float:
        path = graph.weighted_shortest_path(a, b, weather, blocked, process_nodes)
        if not path:
            return float('inf')
        return sum(graph.edge_cost(path[i], path[i+1], weather, blocked, process_nodes)
                   for i in range(len(path)-1))

    direct = _weighted_path_cost(current, goal)
    via_task = _weighted_path_cost(current, task_node) + _weighted_path_cost(task_node, goal)

    if direct == float('inf') or via_task == float('inf'):
        return 999

    # Normalize to approximate frame cost (divide by 1000 to get ~frame units)
    return int((via_task - direct) / 1000)


def _handle_resources(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph, current_node_id: str,
    current_node: dict | None, phase: str, weather: dict | None,
) -> dict | None:
    """Handle resource claiming strategy (策略文档 §6).

    Returns action dict or None.
    """
    if current_node is None:
        return None
    if phase == "RUSH" or round_num >= 360:
        return None

    resources = find_available_resources(current_node)
    if not resources:
        return None

    my_resources = get_player_resources(player)

    # Filter to only high-value resources worth claiming
    HIGH_VALUE_RESOURCES = {"FAST_HORSE", "SHORT_HORSE", "ICE_BOX"}
    WINDOW_RESOURCES = {"OFFICIAL_PERMIT", "PASS_TOKEN"}
    # GUARD_RESERVE_FOR_GATE: reserve 1 permit for gate contest (策略文档 §15)
    PERMIT_RESERVE = 1

    for rtype, count in resources:
        # Skip if already have this resource
        if my_resources.get(rtype, 0) >= 1 and rtype in HIGH_VALUE_RESOURCES:
            continue
        # Only claim high-value resources (FAST_HORSE, SHORT_HORSE, ICE_BOX)
        if rtype in HIGH_VALUE_RESOURCES:
            logger.info("Round %d: Claiming resource %s at %s", round_num, rtype, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
        # Claim OFFICIAL_PERMIT/PASS_TOKEN for window contests
        # Keep at least PERMIT_RESERVE+1 (1 for current use + reserve for GATE)
        if rtype in WINDOW_RESOURCES:
            total_permits = my_resources.get("OFFICIAL_PERMIT", 0) + my_resources.get("PASS_TOKEN", 0)
            if total_permits < PERMIT_RESERVE + 1:
                logger.info("Round %d: Claiming resource %s at %s (for window contests)", round_num, rtype, current_node_id)
                return make_action(match_id, round_num, player_id, [
                    make_claim_resource_action(current_node_id, rtype)
                ])
            continue
        # Claim BOAT_RIGHT (策略文档 §6.1: 仅领取, passive)
        if rtype == "BOAT_RIGHT" and my_resources.get("BOAT_RIGHT", 0) < 1:
            logger.info("Round %d: Claiming BOAT_RIGHT at %s", round_num, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
        # 不领取 INTEL：路线距离常超限，改用小分队探路

    return None


def _handle_force_delivery_resource(
    match_id: str,
    round_num: int,
    player_id: int,
    player: dict,
    graph: MapGraph,
    current_node_id: str,
    current_node: dict | None,
    gate_node_id: str,
    terminal_node_ids: list[str],
    weather: dict | None,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    inquire_nodes: list[dict],
) -> dict | None:
    """Claim only resources that directly shorten the forced delivery route."""
    if current_node is None or has_resource(player, "FAST_HORSE"):
        return None
    direct_target = _find_direct_delivery_step(
        graph, current_node_id, player, gate_node_id, terminal_node_ids,
        weather, process_nodes, processed_node_ids,
    )
    if not direct_target:
        return None
    target_node = find_node_by_id(inquire_nodes, direct_target)
    approaching_key_pass = _is_approaching_key_pass(
        current_node_id, direct_target, inquire_nodes, graph,
    )
    long_road = _edge_distance(graph, current_node_id, direct_target) >= 30
    if not approaching_key_pass and not (long_road and is_key_pass_node(target_node)):
        return None
    for rtype, _count in find_available_resources(current_node):
        if rtype in ("FAST_HORSE", "SHORT_HORSE"):
            logger.info("Round %d: FORCE_DELIVERY claiming %s at %s", round_num, rtype, current_node_id)
            return make_action(match_id, round_num, player_id, [
                make_claim_resource_action(current_node_id, rtype)
            ])
    return None


def _pick_squad_scout_target(
    graph: MapGraph,
    current_node_id: str,
    gate_node_id: str,
    terminal_node_ids: list[str],
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    visited_node_ids: set[str],
    scouted_node_ids: set[str],
    weather: dict | None,
    player: dict,
    obstacle_nodes: set[str] | None = None,
) -> str | None:
    """沿主路线下一未探路的固定处理站派出小分队。

    与 _find_move_target 共用 _get_goal_node 与加权最短路，不写死任何节点编号。
    探路标记使处理帧 -3（任务书 §6.4.1），优先覆盖主车队即将到达的处理站。
    """
    if not current_node_id:
        return None
    remaining = _remaining_fixed_process_nodes(process_nodes, processed_node_ids)
    if not remaining:
        return None
    if get_squad_count(player) < 2:
        return None
    if get_task_score(player) >= TASK_SCORE_STRETCH:
        return None

    goal = _get_goal_node(
        player, gate_node_id, terminal_node_ids, graph,
        current_node_id, weather, None, remaining,
    )
    if not goal:
        return None

    path_blocked = set(obstacle_nodes or [])
    path_blocked.update(visited_node_ids)

    path = graph.weighted_shortest_path(
        current_node_id, goal, weather, path_blocked, remaining,
    )
    if path:
        for nid in path[1:]:
            if nid in scouted_node_ids or nid in visited_node_ids:
                continue
            if nid in remaining:
                return nid

    # 分支地图：主路径上无候选时，选路线累计距离最近的未探路处理站
    best_node = None
    best_dist = float("inf")
    for nid in remaining:
        if nid in scouted_node_ids or nid in visited_node_ids:
            continue
        dist = graph.route_distance(current_node_id, nid)
        if dist <= 0 or dist >= best_dist:
            continue
        best_dist = dist
        best_node = nid
    return best_node


def _find_intel_target(
    graph: MapGraph,
    current_node_id: str,
    process_nodes: dict[str, dict] | None,
    processed_node_ids: set[str],
    visited_node_ids: set[str],
    failed_intel_targets: set[str] | None = None,
) -> str | None:
    """Find next unprocessed process node within INTEL route distance limit."""
    remaining = _remaining_fixed_process_nodes(process_nodes, processed_node_ids)
    if not remaining:
        return None
    if failed_intel_targets is None:
        failed_intel_targets = set()
    best_node = None
    best_dist = float("inf")
    for nid in remaining:
        if nid in visited_node_ids:
            continue
        if nid in failed_intel_targets:
            continue
        dist = graph.route_distance(current_node_id, nid)
        if dist <= INTEL_MAX_DISTANCE and dist < best_dist:
            best_dist = dist
            best_node = nid
    return best_node


def _handle_use_resources(
    match_id: str, round_num: int, player_id: int,
    player: dict, current_node_id: str, graph: MapGraph,
    weather: dict | None, phase: str,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    failed_intel_targets: set[str] | None = None,
    last_move_failed: bool = False,
    last_move_error: str = "",
) -> dict | None:
    """Handle using resources: ice box, horses, intel (策略文档 §6.1)."""
    if processed_node_ids is None:
        processed_node_ids = set()
    if visited_node_ids is None:
        visited_node_ids = set()
    if failed_intel_targets is None:
        failed_intel_targets = set()
    freshness = get_freshness(player)
    force_delivery = _should_force_delivery(round_num, phase, player)

    # Use INTEL only when route distance ≤15 and target not blacklisted
    if (
        has_resource(player, "INTEL")
        and not force_delivery
        and not (last_move_failed and last_move_error == "TARGET_NOT_REACHABLE")
    ):
        intel_target = _find_intel_target(
            graph, current_node_id, process_nodes, processed_node_ids,
            visited_node_ids, failed_intel_targets,
        )
        if intel_target:
            logger.info("Round %d: Using INTEL on %s (route_dist=%d)", round_num, intel_target,
                        graph.route_distance(current_node_id, intel_target))
            return make_action(match_id, round_num, player_id, [
                make_use_resource_action("INTEL", intel_target)
            ])

    # Use ICE_BOX when freshness is low or preemptively before bad weather/routes
    # (策略文档 §6.1: 鲜度<72 或酷暑/山路前)
    if has_resource(player, "ICE_BOX"):
        use_ice = False
        if force_delivery and freshness >= 20:
            use_ice = False
        elif freshness < ICE_BOX_FRESHNESS_THRESHOLD:
            use_ice = True
        # Preemptive: check if next route segment is mountain or hot weather
        elif weather and freshness < 80:
            forecasts = weather.get("forecast", [])
            for fw in forecasts:
                wtype = fw.get("type", "")
                if wtype == "HOT":
                    use_ice = True
                    break
        if not use_ice:
            # Check if next step goes through mountain
            neighbors = graph.get_neighbors(current_node_id)
            for n in neighbors:
                if graph.get_edge_route_type(current_node_id, n) == "MOUNTAIN" and freshness < 80:
                    use_ice = True
                    break
        if use_ice:
            logger.info("Round %d: Using ICE_BOX (freshness=%.1f)", round_num, freshness)
            return make_action(match_id, round_num, player_id, [
                make_use_resource_action("ICE_BOX")
            ])

    # Save horse buffs for forced delivery; using them mid-route wastes the short duration.
    if force_delivery and has_resource(player, "FAST_HORSE"):
        neighbors = graph.get_neighbors(current_node_id)
        for n in neighbors:
            if graph.get_edge_route_type(current_node_id, n) == "ROAD":
                logger.info("Round %d: Using FAST_HORSE before road move", round_num)
                return make_action(match_id, round_num, player_id, [
                    make_use_resource_action("FAST_HORSE")
                ])

    if force_delivery and has_resource(player, "SHORT_HORSE") and not has_resource(player, "FAST_HORSE"):
        neighbors = graph.get_neighbors(current_node_id)
        if neighbors:
            logger.info("Round %d: Using SHORT_HORSE before move", round_num)
            return make_action(match_id, round_num, player_id, [
                make_use_resource_action("SHORT_HORSE")
            ])

    return None


def _handle_combat(
    match_id: str, round_num: int, player_id: int,
    player: dict, graph: MapGraph,
    current_node_id: str, gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None,
    mode: str, phase: str, inquire_nodes: list[dict],
    opp_player: dict | None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    visited_node_ids: set[str] | None = None,
    my_team_id: str = "",
    bounties: list[dict] | None = None,
) -> dict | None:
    """Handle combat: guard, break, squad (策略文档 §8)."""
    if obstacle_nodes is None:
        obstacle_nodes = set()
    if process_nodes is None:
        process_nodes = {}
    if visited_node_ids is None:
        visited_node_ids = set()
    if bounties is None:
        bounties = []
    if not my_team_id:
        my_team_id = get_team_id(player)

    # 优先攻坚有悬赏的敌方设卡 (任务书 §6.3.3)
    bounty_node = _find_bounty_node(bounties, my_team_id)
    if bounty_node and bounty_node in graph.get_neighbors(current_node_id):
        for node in inquire_nodes:
            if node.get("nodeId") != bounty_node:
                continue
            guard = node.get("guard", {})
            if is_enemy_guard(guard, my_team_id, player_id):
                good = min(get_good_fruit(player), 2)
                bad = min(get_bad_fruit(player), 2)
                if good + bad > 0:
                    action = make_break_guard_action(bounty_node, good_fruit=good, bad_fruit=bad)
                    if phase == "RUSH" and (bad >= 2 or good >= 1):
                        action["rushTactic"] = "BREAK_ORDER"
                    logger.info("Round %d: Breaking bounty guard at %s", round_num, bounty_node)
                    return make_action(match_id, round_num, player_id, [action])

    # SET_GUARD: 领先时可设卡钓鱼；宫门/关键关隘争夺时设卡
    should_set_guard = (
        mode == "GATE_FIGHT"
        or (phase == "RUSH" and gate_node_id and current_node_id == gate_node_id)
        or (mode == "STEADY" and get_good_fruit(player) >= 2)
    )
    if should_set_guard and get_good_fruit(player) >= 1:
        guard_target = _find_guard_target(
            graph, current_node_id, gate_node_id, terminal_node_ids,
            weather, blocked, player, inquire_nodes, my_team_id,
        )
        if guard_target and guard_target == current_node_id:
            logger.info("Round %d: Setting guard at current node %s", round_num, guard_target)
            extra = 1 if get_good_fruit(player) >= 2 else 0
            return make_action(match_id, round_num, player_id, [
                make_set_guard_action(guard_target, extra_good_fruit=extra)
            ])

    # --- BREAK_GUARD with optional BREAK_ORDER (策略文档 §8.2, §10) ---
    goal = gate_node_id or (terminal_node_ids[0] if terminal_node_ids else "")
    if goal and blocked:
        optimal_path = graph.shortest_path(current_node_id, goal, weather, obstacle_nodes)
        if optimal_path and len(optimal_path) >= 2:
            next_hop = optimal_path[1]
            if next_hop in blocked:
                for node in inquire_nodes:
                    if node.get("nodeId") == next_hop:
                        guard = node.get("guard", {})
                        if is_enemy_guard(guard, my_team_id, player_id):
                            good = min(get_good_fruit(player), 2)
                            bad = min(get_bad_fruit(player), 2)
                            if good + bad > 0:
                                action = make_break_guard_action(next_hop, good_fruit=good, bad_fruit=bad)
                                # Bind BREAK_ORDER if in RUSH and have resources (策略文档 §10: +3攻坚)
                                if phase == "RUSH" and (bad >= 2 or good >= 1):
                                    action["rushTactic"] = "BREAK_ORDER"
                                    logger.info("Round %d: Breaking guard at %s with BREAK_ORDER", round_num, next_hop)
                                else:
                                    logger.info("Round %d: Breaking guard at %s (blocking path)", round_num, next_hop)
                                return make_action(match_id, round_num, player_id, [action])
                        # Try FORCED_PASS instead
                        logger.info("Round %d: Forced pass at %s", round_num, next_hop)
                        return make_action(match_id, round_num, player_id, [
                            make_forced_pass_action(next_hop)
                        ])

    # --- Squad actions (策略文档 §8.4) — only if not RUSH ---
    if phase != "RUSH":
        squad_count = get_squad_count(player)

        # SQUAD_CLEAR: Clear obstacles without main team (策略文档 §8.4: 2人手)
        if squad_count >= 8:
            for node in inquire_nodes:
                if node.get("hasObstacle", False) and node.get("nodeId") != current_node_id:
                    nid = node.get("nodeId", "")
                    # Check if obstacle is on our path
                    if goal:
                        path = graph.shortest_path(current_node_id, goal, weather, obstacle_nodes)
                        if path and nid in path:
                            logger.info("Round %d: Squad clear at %s", round_num, nid)
                            return make_action(match_id, round_num, player_id, [
                                make_squad_clear_action(nid)
                            ])

        # SQUAD_REINFORCE: Reinforce our own guard at key nodes (策略文档 §8.4: 2人手)
        if squad_count >= 8:
            for node in inquire_nodes:
                guard = node.get("guard", {})
                owner_team = guard.get("ownerTeamId") if guard else ""
                if (guard and owner_team == my_team_id
                        and guard_is_active(guard)
                        and node.get("nodeId") != current_node_id):
                    nid = node.get("nodeId", "")
                    logger.info("Round %d: Squad reinforce at %s", round_num, nid)
                    return make_action(match_id, round_num, player_id, [
                        make_squad_reinforce_action(nid)
                    ])

        # SQUAD_WEAKEN: Weaken enemy guard (策略文档 §8.4: 2人手, 性价比高)
        if squad_count >= 2 and opp_player:
            for node in inquire_nodes:
                guard = node.get("guard", {})
                if (is_enemy_guard(guard, my_team_id, player_id)
                        and node.get("nodeId") != current_node_id):
                    nid = node.get("nodeId", "")
                    logger.info("Round %d: Squad weaken at %s", round_num, nid)
                    return make_action(match_id, round_num, player_id, [
                        make_squad_weaken_action(nid)
                    ])

    return None


def _find_guard_target(
    graph: MapGraph, current_node_id: str,
    gate_node_id: str, terminal_node_ids: list[str],
    weather: dict | None, blocked: set[str] | None,
    player: dict, inquire_nodes: list[dict],
    my_team_id: str,
) -> str | None:
    """Find a good node to set guard on (策略文档 §8.1).

    Key: don't set guard on our own route. Target opponent's likely route.
    """
    # Find nodes that are NOT on our path to gate/terminal
    goal = _get_goal_node(player, gate_node_id, terminal_node_ids, graph, current_node_id, weather, blocked)
    if not goal:
        return None

    our_path = graph.weighted_shortest_path(current_node_id, goal, weather, blocked)
    if our_path and current_node_id in our_path[-3:]:
        return None

    neighbors = graph.get_neighbors(current_node_id)
    if len(neighbors) <= 3:
        for node in inquire_nodes:
            if node.get("nodeId") == current_node_id and not guard_is_active(node.get("guard")):
                return current_node_id

    return None


def _handle_rush_tactics(
    match_id: str, round_num: int, player_id: int,
    player: dict, current_node_id: str, phase: str, mode: str,
    graph: MapGraph | None = None,
    gate_node_id: str = "",
    terminal_node_ids: list[str] | None = None,
    weather: dict | None = None,
    obstacle_nodes: set[str] | None = None,
    process_nodes: dict[str, dict] | None = None,
    processed_node_ids: set[str] | None = None,
    visited_node_ids: set[str] | None = None,
    rush_speed_failed: bool = False,
) -> dict | None:
    """Handle rush tactics: RUSH_SPEED, RUSH_PROTECT (策略文档 §10).

    Only available after RUSH phase. Each can be used once per match.
    RUSH_SPEED与马互斥: 有马buff时不使用.
    """
    if phase != "RUSH":
        return None

    # RUSH_SPEED can only be used when IDLE
    state = player.get("state", "")
    if state != "IDLE":
        return None

    freshness = get_freshness(player)

    # RUSH_PROTECT: 鲜度<50, 停靠节点使用 (策略文档 §10: 0成本)
    if freshness < RUSH_PROTECT_FRESHNESS:
        logger.info("Round %d: Using RUSH_PROTECT (freshness=%.1f)", round_num, freshness)
        return make_action(match_id, round_num, player_id, [make_rush_protect_action()])

    # RUSH_SPEED: 无马、鲜度尚可、有好果、抢用时分 (任务书 §6.5)
    if (
        not rush_speed_failed
        and not _has_active_speed_buff(player)
        and not has_resource(player, "FAST_HORSE")
        and not has_resource(player, "SHORT_HORSE")
        and get_good_fruit(player) >= 2
        and freshness >= 40
        and mode in ("RACE", "GATE_FIGHT", "AGGRESSIVE")
    ):
        logger.info("Round %d: Using RUSH_SPEED (freshness=%.1f)", round_num, freshness)
        return make_action(match_id, round_num, player_id, [make_rush_speed_action()])

    return None
