import jax
import jax.numpy as jnp
import jax.random as jr
import jax.debug as jdebug

from typing import Optional, Dict, Tuple
import functools as ft

from dgppo.utils.graph import EdgeBlock
from dgppo.utils.typing import Action, Array, Pos2d, Reward, State
from dgppo.env.lidar_env.base import (
    LidarEnv, LidarEnvState, LidarEnvGraphsTuple, TARGET_TERRAIN_ID,
    get_terrain_id, _terrain_from_perp,
)
from dgppo.env.utils import get_ray_alphas
from dgppo.utils.utils import jax_vmap, merge01

DELAY_STEPS = 15  # round(0.5 s / (1/30 s)) — 500ms action delay

ALL_POSSIBLE_REGION_NAMES = [
        "open_space",
        "approach_bridge_0",
        "on_bridge_0",
        "exit_bridge_0"
    ]

def _calculate_bearing_reward(agent_vel: jnp.ndarray, target_bearing: float) -> float:
    target_vector = jnp.array([jnp.cos(target_bearing), jnp.sin(target_bearing)])
    norm_agent_vel = jnp.linalg.norm(agent_vel)
    safe_norm_agent_vel = jnp.where(norm_agent_vel > 1e-6, norm_agent_vel, 1e-6)
    return jnp.dot(agent_vel, target_vector) / safe_norm_agent_vel  # direction only

def _calculate_cluster_reward_per_agent(
    current_cluster_oh_episode_i: Array,
    start_cluster_oh_episode_i: Array,
    next_cluster_oh_episode_i: Array,
    has_been_awarded: jnp.ndarray,
    # Reward coefficients
    next_cluster_bonus: float,
    stay_in_cluster_bonus: float,
    incorrect_cluster_penalty: float

) -> float:

    current_cluster_id_episode_i = jnp.argmax(current_cluster_oh_episode_i)
    start_cluster_id_episode_i = jnp.argmax(start_cluster_oh_episode_i)
    next_cluster_id_episode_i = jnp.argmax(next_cluster_oh_episode_i)

    current_agent_reward = 0.0

    has_moved_into_next_cluster = (current_cluster_id_episode_i == next_cluster_id_episode_i) & \
                                 (current_cluster_id_episode_i != start_cluster_id_episode_i)

    # `give_bonus` is true only on the first step the agent moves into the next cluster
    give_bonus = has_moved_into_next_cluster & (~has_been_awarded)

    # Award the one-time bonus correctly
    bonus_reward = jnp.where(give_bonus, next_cluster_bonus, 0.0)

    # Update the flag to prevent future bonuses
    updated_has_been_awarded = jnp.logical_or(has_been_awarded, give_bonus)

    current_agent_reward += bonus_reward

    # This is the line that was missing. It awards a continuous bonus for staying in the target cluster.
    is_in_next_cluster = (current_cluster_id_episode_i == next_cluster_id_episode_i)
    current_agent_reward += jnp.where(is_in_next_cluster, stay_in_cluster_bonus, 0.0)

    is_incorrect_path_taken = (current_cluster_id_episode_i != start_cluster_id_episode_i) & \
                             (current_cluster_id_episode_i != next_cluster_id_episode_i)

    current_agent_reward += jnp.where(
        is_incorrect_path_taken,
        incorrect_cluster_penalty,
        0.0
    )

    return current_agent_reward, updated_has_been_awarded


def _terrain_reward_per_agent(
    boundary_hit_positions: jnp.ndarray,   # (n_rays, 2) — boundary hit positions
    boundary_terrain_ids: jnp.ndarray,     # (n_rays,)   — terrain on OTHER SIDE of each boundary
    agent_pos: jnp.ndarray,                # (2,)
    current_terrain_id: jnp.ndarray,       # scalar
    sense_range: float,
) -> float:
    """
    Mid-range terrain navigation reward using per-ray delta terrain value.

    For every boundary hit ray j:
        contribution_j = proximity_j × (V_other_j − V_current)

    where V = [Road: −1.0, Grass: −0.3, Sidewalk: +1.0] and
          proximity_j = max(0, 1 − dist_j / sense_range)   (0 when ray hits sensor boundary)

    This naturally rewards:
      − approaching sidewalk from road/grass      (+1.3 / +1.7 per ray)
      − staying far from road boundary on sidewalk  (proximity → 0 when far from edge)
      − approaching road from sidewalk or grass    (−2.0 / −0.7 per ray, penalised)
    Rays with no visible boundary (alpha=2.0 → dist > sense_range) contribute 0.
    """
    TERRAIN_VALUES = jnp.array([-1.0, -0.3, 1.0])  # Road=0, Grass=1, Sidewalk=2

    v_current = TERRAIN_VALUES[current_terrain_id]          # scalar
    v_other   = TERRAIN_VALUES[boundary_terrain_ids]        # (n_rays,)
    delta_v   = v_other - v_current                         # (n_rays,)

    dists     = jnp.linalg.norm(boundary_hit_positions - agent_pos[None, :], axis=-1)  # (n_rays,)
    proximity = jnp.where(dists < sense_range, 1.0 - dists / sense_range, 0.0)         # (n_rays,)

    on_target = (current_terrain_id == TARGET_TERRAIN_ID)
    return jnp.where(on_target, 0.0, (delta_v * proximity).mean())


def _calculate_preference_vector_reward(
    boundary_hit_positions: jnp.ndarray,  # (n_rays, 2) — boundary hit positions
    boundary_terrain_ids: jnp.ndarray,    # (n_rays,)   — terrain on OTHER SIDE
    agent_pos: jnp.ndarray,               # (2,)
    agent_vel: jnp.ndarray,               # (2,)
    current_terrain_id: jnp.ndarray,      # scalar
    target_bearing: jnp.ndarray,          # scalar — global waypoint bearing (fallback)
    sense_range: float,
) -> float:
    """
    Road-embedding preference vector reward for one agent.

    Outside target terrain:
        cos(vel, direction to nearest target boundary), or global bearing fallback.

    Inside target terrain (anti-parallel boundary pairing):
        Edge A = nearest non-target boundary.
        Edge B = non-target boundary most anti-parallel to A (opposite side of strip).
        road_dir = perp(B − A), sign chosen so it agrees with global bearing.
        Blends cosine_road with a centering pull (cosine toward A–B midpoint) scaled
        by the agent's normalised lateral offset from the strip centre.
        Falls back to global bearing when A/B are not both visible.
    """
    dists = jnp.linalg.norm(boundary_hit_positions - agent_pos[None, :], axis=-1)  # (n_rays,)
    in_target = (current_terrain_id == TARGET_TERRAIN_ID)

    vel_norm = jnp.linalg.norm(agent_vel)
    safe_vel_norm = jnp.where(vel_norm > 1e-6, vel_norm, 1e-6)

    # Global bearing fallback (always well-defined)
    bearing_vec = jnp.array([jnp.cos(target_bearing), jnp.sin(target_bearing)])
    cosine_bearing = jnp.dot(agent_vel, bearing_vec) / safe_vel_norm

    # ── Not in target: point toward nearest entry ─────────────────────────
    is_entry = (boundary_terrain_ids == TARGET_TERRAIN_ID)
    entry_masked = jnp.where(is_entry, dists, sense_range * 2.0)
    nearest_entry_idx = jnp.argmin(entry_masked)
    any_entry_visible = entry_masked[nearest_entry_idx] < sense_range

    entry_dir = boundary_hit_positions[nearest_entry_idx] - agent_pos
    entry_dir_norm = entry_dir / (jnp.linalg.norm(entry_dir) + 1e-6)
    cosine_entry = jnp.where(
        any_entry_visible,
        jnp.dot(agent_vel, entry_dir_norm) / safe_vel_norm,
        cosine_bearing,
    )

    # ── In target: anti-parallel boundary pairing ─────────────────────────
    is_non_target = (boundary_terrain_ids != TARGET_TERRAIN_ID)
    non_target_dists = jnp.where(is_non_target, dists, sense_range * 2.0)

    # Edge A: nearest non-target boundary
    idx_a = jnp.argmin(non_target_dists)
    hit_a = boundary_hit_positions[idx_a]
    dist_a = non_target_dists[idx_a]

    dir_a = hit_a - agent_pos
    dir_a_norm = dir_a / (jnp.linalg.norm(dir_a) + 1e-6)

    # Edge B: non-target boundary most anti-parallel to A
    dirs = boundary_hit_positions - agent_pos[None, :]                         # (n_rays, 2)
    dirs_norm = dirs / (jnp.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-6)
    dot_with_a = dirs_norm @ dir_a_norm                                        # (n_rays,)
    anti_score = jnp.where(is_non_target, dot_with_a, 1.0)
    anti_score = anti_score.at[idx_a].set(1.0)  # exclude A itself

    idx_b = jnp.argmin(anti_score)
    hit_b = boundary_hit_positions[idx_b]
    dist_b = dists[idx_b]

    # Gate: both visible and B sufficiently anti-parallel
    any_both_visible = (dist_a < sense_range) & (dist_b < sense_range) & (anti_score[idx_b] < -0.1)

    # Road direction = perp(B − A), signed to agree with global bearing
    cross = hit_b - hit_a
    road_opt = jnp.array([-cross[1], cross[0]])
    road_dir = jnp.where(jnp.dot(road_opt, bearing_vec) >= 0, road_opt, -road_opt)
    road_dir_norm = road_dir / (jnp.linalg.norm(road_dir) + 1e-6)
    cosine_road = jnp.dot(agent_vel, road_dir_norm) / safe_vel_norm

    # Centering blend: pull toward A–B midpoint proportional to lateral offset
    midpoint = (hit_a + hit_b) / 2.0
    lateral_vec = midpoint - agent_pos
    lateral_norm = lateral_vec / (jnp.linalg.norm(lateral_vec) + 1e-6)
    cosine_lateral = jnp.dot(agent_vel, lateral_norm) / safe_vel_norm

    strip_half_width = jnp.linalg.norm(cross) / 2.0 + 1e-6
    offset = jnp.clip(jnp.linalg.norm(lateral_vec) / strip_half_width, 0.0, 1.0)

    cosine_road_blended = (1.0 - 0.5 * offset) * cosine_road + 0.5 * offset * cosine_lateral
    cosine_center = jnp.where(any_both_visible, cosine_road_blended, cosine_bearing)

    return jnp.where(in_target, cosine_center, cosine_entry)


class LidarTarget(LidarEnv):

    COSINE_SIM_REWARD_COEFF = 0.2
    NEXT_CLUSTER_BONUS = 40.0
    INCORRECT_CLUSTER_PENALTY = -2.0
    STAY_IN_CLUSTER_BONUS = 0.5
    VELOCITY_PENALTY_IN_CLUSTER = -0.1
    TERRAIN_REWARD_COEFF = 0.0
    PREF_VECTOR_REWARD_COEFF = 0.0       # disabled: matches 3c8a670 proven training setup
    TERRAIN_PENALTY_COEFF = 0.5
    ROAD_SPEED_COEFF = 0.05
    GRASS_SPEED_COEFF = 0.1

    PARAMS = {
        "car_radius": 0.02,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 2,
        "default_area_size": 1.5,
        "dist2goal": 0.01, # This value is now largely unused for reward unless repurposed
        "top_k_rays": 8,

        # Ensure bridge-related params are also in LidarEnv's PARAMS for reset() to use
        "num_bridges": 1,
        "bridge_length_range": [0.5, 1.0],
        "bridge_gap_width_range": [0.2, 0.4],
        "bridge_wall_thickness_range": [0.05, 0.1],
        "bridge_bend_angle_range": [-0.4, 0.4]   # ≈ ±23° bend — POC angular bridge
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None
    ):
        area_size = LidarTarget.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarTarget, self).__init__(num_agents, area_size, max_step, dt, params)

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Tuple[Reward, jnp.ndarray]:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        agent_vel = agent_states[:, 2:4]

        # --- Extract necessary info from env_states ---
        env_states = graph.env_states
        current_cluster_oh_episode = env_states.current_cluster_oh # Per-agent current high-level cluster OH
        start_cluster_oh_episode = env_states.start_cluster_oh
        next_cluster_oh_episode = env_states.next_cluster_oh     # Per-agent next high-level cluster OH
        actual_cluster_id = jnp.argmax(current_cluster_oh_episode, axis=-1)
        bonus_awarded_state = env_states.next_cluster_bonus_awarded

        bridge_center = env_states.bridge_center
        bridge_length = env_states.bridge_length
        bridge_gap_width = env_states.bridge_gap_width
        bridge_wall_thickness = env_states.bridge_wall_thickness
        bridge_theta = env_states.bridge_theta # In radians

        bearing = env_states.bearing
        current_terrain_oh = env_states.current_terrain_oh  # (n_agents, 3)
        current_terrain_id = jnp.argmax(current_terrain_oh, axis=-1)  # (n_agents,)

        # Check if the agent is in the next cluster
        next_cluster_id = jnp.argmax(next_cluster_oh_episode, axis=-1)
        is_in_next_cluster = (actual_cluster_id == next_cluster_id)

        # ── Global waypoint bearing reward (long-range, cluster navigation) ──
        dense_reward = jnp.where(
            ~is_in_next_cluster,  # Condition: If NOT in the next cluster
            jax_vmap(_calculate_bearing_reward, in_axes=(0, 0))(agent_vel, bearing).mean(),
            0.0 # If in the next cluster, the reward is 0.0
        ).mean()
        reward = dense_reward * self.COSINE_SIM_REWARD_COEFF

        is_bridge_env = bridge_length > 0.0

        vmap_cluster_reward_fn = jax_vmap(ft.partial(
            _calculate_cluster_reward_per_agent,
            stay_in_cluster_bonus=self.STAY_IN_CLUSTER_BONUS,
            next_cluster_bonus=self.NEXT_CLUSTER_BONUS,
            incorrect_cluster_penalty=self.INCORRECT_CLUSTER_PENALTY,
        ), in_axes=(0, 0, 0, 0))

        per_agent_reward, per_agent_bonus_awarded_updated = vmap_cluster_reward_fn(
            current_cluster_oh_episode,
            start_cluster_oh_episode,
            next_cluster_oh_episode,
            bonus_awarded_state
        )

        # Use jnp.where separately for each output
        reward_cluster = jnp.where(is_bridge_env, per_agent_reward, 0.0)
        next_cluster_bonus_awarded_updated = jnp.where(is_bridge_env, per_agent_bonus_awarded_updated, bonus_awarded_state)

        reward += jnp.mean(reward_cluster)

        velocity_magnitude_sq = jnp.linalg.norm(agent_vel, axis=1) ** 2
        velocity_penalty = jnp.where(is_in_next_cluster, velocity_magnitude_sq, 0.0).mean()
        reward += velocity_penalty * self.VELOCITY_PENALTY_IN_CLUSTER

        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.001

        # ── Immediate terrain penalties (one-hot "haptic" feedback) ──────────
        # Road is strongly penalised; Grass is mildly penalised; Sidewalk is rewarded.
        road_penalty    = jnp.where(current_terrain_id == 0, -1.0, 0.0)
        grass_penalty   = jnp.where(current_terrain_id == 1, -0.3, 0.0)
        sidewalk_bonus  = jnp.where(current_terrain_id == 2,  0.5, 0.0)
        immediate_terrain_penalty = (road_penalty + grass_penalty + sidewalk_bonus).mean()
        reward += jnp.where(is_bridge_env, immediate_terrain_penalty * self.TERRAIN_PENALTY_COEFF, 0.0)

        # ── Speed penalty: terrain-graded (road=slight, grass=more, sidewalk=none) ──
        speed_sq = jnp.linalg.norm(agent_vel, axis=1) ** 2
        road_speed_pen  = jnp.where(current_terrain_id == 0, speed_sq * self.ROAD_SPEED_COEFF,  0.0)
        grass_speed_pen = jnp.where(current_terrain_id == 1, speed_sq * self.GRASS_SPEED_COEFF, 0.0)
        terrain_speed_penalty = (road_speed_pen + grass_speed_pen).mean()
        reward -= jnp.where(is_bridge_env, terrain_speed_penalty, 0.0)

        # ── Semantic lidar rewards (mid-range + long-range from lidar data) ──
        # is_bridge_env is a JAX array — no Python if; gate with jnp.where per term.
        n_rays = self._params["n_rays"]
        sense_range = self._params["comm_radius"]

        # Use full lidar data from env_states (all n_rays per agent, not top-k graph subset)
        all_hit_pos = jnp.reshape(
            env_states.lidar_hit_positions[:2 * n_rays * self.num_agents],
            (self.num_agents, 2 * n_rays, 2),
        )
        boundary_hit_pos = all_hit_pos[:, n_rays:, :]  # (n_agents, n_rays, 2)

        all_terrain_ids = jnp.reshape(
            env_states.lidar_hit_terrain_ids[:2 * n_rays * self.num_agents],
            (self.num_agents, 2 * n_rays),
        )
        boundary_terrain_ids = all_terrain_ids[:, n_rays:]  # (n_agents, n_rays)

        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # Mid-range: per-ray delta terrain value reward
        per_agent_terrain_reward = jax_vmap(
            ft.partial(_terrain_reward_per_agent, sense_range=sense_range)
        )(boundary_hit_pos, boundary_terrain_ids, agent_pos, current_terrain_id)
        reward += jnp.where(is_bridge_env, jnp.mean(per_agent_terrain_reward) * self.TERRAIN_REWARD_COEFF, 0.0)

        # Long-range: preference vector (sidewalk entry / centerline tracking)
        # Falls back to global bearing when terrain boundaries are not visible.
        per_agent_pref_reward = jax_vmap(
            ft.partial(_calculate_preference_vector_reward, sense_range=sense_range),
            in_axes=(0, 0, 0, 0, 0, 0),
        )(boundary_hit_pos, boundary_terrain_ids, agent_pos, agent_vel, current_terrain_id, bearing)
        reward += jnp.where(is_bridge_env, per_agent_pref_reward.mean() * self.PREF_VECTOR_REWARD_COEFF, 0.0)

        return reward*0.1, next_cluster_bonus_awarded_updated

    def get_reward_components(self, graph: LidarEnvGraphsTuple) -> dict:
        """Returns individual reward components as a flat dict for wandb logging.
        Call outside JIT (trainer eval loop) on a single graph."""
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        env_states = graph.env_states
        agent_vel = agent_states[:, 2:4]
        bearing = env_states.bearing
        current_terrain_oh = env_states.current_terrain_oh
        current_terrain_id = jnp.argmax(current_terrain_oh, axis=-1)
        is_bridge_env = env_states.bridge_length > 0.0

        actual_cluster_id = jnp.argmax(env_states.current_cluster_oh, axis=-1)
        next_cluster_id = jnp.argmax(env_states.next_cluster_oh, axis=-1)
        is_in_next_cluster = (actual_cluster_id == next_cluster_id)

        # Bearing reward
        bearing_r = jnp.where(
            ~is_in_next_cluster,
            jax_vmap(_calculate_bearing_reward, in_axes=(0, 0))(agent_vel, bearing).mean(),
            0.0,
        ).mean() * self.COSINE_SIM_REWARD_COEFF

        # Immediate terrain penalties
        road_penalty  = jnp.where(current_terrain_id == 0, -1.0, 0.0)
        grass_penalty = jnp.where(current_terrain_id == 1, -0.3, 0.0)
        immediate_r = float(jnp.where(
            is_bridge_env,
            (road_penalty + grass_penalty).mean() * self.TERRAIN_PENALTY_COEFF,
            0.0,
        ))

        # Speed penalty on wrong terrain (terrain-graded)
        speed_sq = jnp.linalg.norm(agent_vel, axis=1) ** 2
        road_speed_pen  = jnp.where(current_terrain_id == 0, speed_sq * self.ROAD_SPEED_COEFF,  0.0)
        grass_speed_pen = jnp.where(current_terrain_id == 1, speed_sq * self.GRASS_SPEED_COEFF, 0.0)
        speed_r = -float(jnp.where(
            is_bridge_env,
            (road_speed_pen + grass_speed_pen).mean(),
            0.0,
        ))

        # Semantic lidar components
        n_rays = self._params["n_rays"]
        sense_range = self._params["comm_radius"]
        all_hit_pos = jnp.reshape(
            env_states.lidar_hit_positions[: 2 * n_rays * self.num_agents],
            (self.num_agents, 2 * n_rays, 2),
        )
        boundary_hit_pos = all_hit_pos[:, n_rays:, :]
        all_terrain_ids = jnp.reshape(
            env_states.lidar_hit_terrain_ids[: 2 * n_rays * self.num_agents],
            (self.num_agents, 2 * n_rays),
        )
        boundary_terrain_ids = all_terrain_ids[:, n_rays:]
        agent_pos = agent_states[:, :2]

        per_agent_terrain = jax_vmap(
            ft.partial(_terrain_reward_per_agent, sense_range=sense_range)
        )(boundary_hit_pos, boundary_terrain_ids, agent_pos, current_terrain_id)
        terrain_r = float(jnp.where(
            is_bridge_env, jnp.mean(per_agent_terrain) * self.TERRAIN_REWARD_COEFF, 0.0
        ))

        per_agent_pref = jax_vmap(
            ft.partial(_calculate_preference_vector_reward, sense_range=sense_range),
            in_axes=(0, 0, 0, 0, 0, 0),
        )(boundary_hit_pos, boundary_terrain_ids, agent_pos, agent_vel, current_terrain_id, bearing)
        pref_r = float(jnp.where(
            is_bridge_env, per_agent_pref.mean() * self.PREF_VECTOR_REWARD_COEFF, 0.0
        ))

        # Current terrain distribution
        terrain_names = {0: "road", 1: "grass", 2: "sidewalk"}
        terrain_fracs = {
            f"eval/terrain_frac_{name}": float((current_terrain_id == tid).mean())
            for tid, name in terrain_names.items()
        }

        # Cost components: agent-agent vs obstacle
        cost = self.get_cost(graph)  # (n_agents, 2): col 0=agent-agent, col 1=obstacle
        cost_agent_agent = float(jnp.maximum(cost[:, 0], 0.0).mean())
        cost_obstacle    = float(jnp.maximum(cost[:, 1], 0.0).mean())

        return {
            "eval/reward_bearing":           float(bearing_r),
            "eval/reward_terrain_immediate": immediate_r,
            "eval/reward_speed_penalty":     speed_r,
            "eval/reward_terrain_midrange":  terrain_r,
            "eval/reward_pref_vector":       pref_r,
            "eval/cost_agent_agent":         cost_agent_agent,
            "eval/cost_obstacle":            cost_obstacle,
            **terrain_fracs,
        }

    def state2feat(self, state: State) -> Array:
        return state

    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        # agent - agent connection
        agent_obs = state.obs_agent if state.obs_agent.shape[0] > 0 else state.agent
        agent_pos = agent_obs[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        edge_feats = (jax_vmap(self.state2feat)(agent_obs)[:, None, :] -
                      jax_vmap(self.state2feat)(agent_obs)[None, :, :])
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(edge_feats, agent_agent_mask, id_agent, id_agent)

        # # agent - goal connection
        agent_goal_edges = []

        # agent - obs connection
        # Each agent has 2*top_k hit nodes in the graph: first top_k = obstacle hits, last top_k = boundary hits
        agent_obs_edges = []
        hits_per_agent = 2 * self._params["top_k_rays"]
        n_hits = hits_per_agent * self.num_agents
        if lidar_data is not None:
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * hits_per_agent, (i + 1) * hits_per_agent)
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, hits_per_agent))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((lidar_feats.shape[0], self.edge_dim - lidar_feats.shape[1]))], axis=-1)
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges


# ---------------------------------------------------------------------------
# Real terrain-boundary lidar (used only by V3/V4)
# ---------------------------------------------------------------------------

def _get_semantic_lidar_single_real(
    agent_pos: jnp.ndarray,              # (2,)
    obstacles,
    bridge_center: jnp.ndarray,          # (2,) — bend/junction point
    bridge_gap_width: jnp.ndarray,
    bridge_wall_thickness: jnp.ndarray,
    bridge_theta: jnp.ndarray,           # segment-1 angle (radians)
    terrain_config: jnp.ndarray,
    num_beams: int,
    sense_range: float,
    bridge_length: jnp.ndarray = 1.0,
    bridge_bend_angle: jnp.ndarray = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-ray semantic lidar with real terrain-boundary detection.
    Handles straight and bent bridges (two-segment).

    For each of B rays returns TWO hit points (2B total):
      [0:B]  — obstacle hits  (or sensor-range endpoint)
      [B:2B] — nearest TARGET_TERRAIN_ID boundary along the ray

    terrain_ids[B:2B] = terrain on the far side of each detected boundary.
    """
    thetas = jnp.linspace(-jnp.pi, jnp.pi - 2 * jnp.pi / num_beams, num_beams)
    dirs   = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)   # (B, 2)
    starts = jnp.tile(agent_pos[None, :], (num_beams, 1))
    ends   = starts + dirs * sense_range

    alphas_obs = get_ray_alphas(starts, ends, obstacles)               # (B,)

    # Target-terrain borders in perpendicular-coordinate space
    half_gap        = bridge_gap_width / 2.0
    full_half       = half_gap + bridge_wall_thickness
    sidewalk_border = bridge_gap_width * 0.2
    road_half       = half_gap - sidewalk_border
    target_borders_c1 = jnp.array([ full_half, -full_half,  full_half, -full_half])
    target_borders_c2 = jnp.array([ road_half, -road_half,  half_gap,  -half_gap])
    target_borders = jnp.where(terrain_config == 1, target_borders_c1, target_borders_c2)  # (4,)

    eps = 1e-8
    dx0 = agent_pos[0] - bridge_center[0]
    dy0 = agent_pos[1] - bridge_center[1]

    def _alphas_for_segment(theta_seg):
        cos_s = jnp.cos(theta_seg);  sin_s = jnp.sin(theta_seg)
        perp_start = -sin_s * dx0 + cos_s * dy0
        perp_dirs  = -sin_s * dirs[:, 0] + cos_s * dirs[:, 1]         # (B,)
        safe_pd    = jnp.where(jnp.abs(perp_dirs) > eps, perp_dirs, eps)
        alphas_b   = (target_borders[None, :] - perp_start) / (safe_pd[:, None] * sense_range)  # (B, 4)
        valid      = (alphas_b > 1e-4) & (alphas_b < 1.0) & (jnp.abs(perp_dirs[:, None]) > eps)
        return jnp.where(valid, alphas_b, 2.0)                         # (B, 4)

    alphas_b_s1 = _alphas_for_segment(bridge_theta)                    # (B, 4)
    alphas_b_s2 = _alphas_for_segment(bridge_theta + bridge_bend_angle)# (B, 4)

    # Both segments contribute; nearest valid boundary wins
    alphas_b_all   = jnp.concatenate([alphas_b_s1, alphas_b_s2], axis=-1)  # (B, 8)
    alphas_terrain = alphas_b_all.min(axis=-1)                              # (B,)

    obs_hits      = agent_pos[None, :] + alphas_obs[:, None]     * sense_range * dirs  # (B, 2)
    boundary_hits = agent_pos[None, :] + alphas_terrain[:, None] * sense_range * dirs  # (B, 2)
    hit_points    = jnp.concatenate([obs_hits, boundary_hits])                          # (2B, 2)

    _get_tid = ft.partial(
        get_terrain_id,
        bridge_center=bridge_center,
        bridge_gap_width=bridge_gap_width,
        bridge_wall_thickness=bridge_wall_thickness,
        bridge_theta=bridge_theta,
        terrain_config=terrain_config,
        bridge_length=bridge_length,
        bridge_bend_angle=bridge_bend_angle,
    )

    obs_terrain_ids = jax_vmap(_get_tid)(obs_hits)                     # (B,)

    BOUNDARY_STEP = 0.005  # step past the boundary to sample the far-side terrain
    boundary_beyond = boundary_hits + BOUNDARY_STEP * dirs             # (B, 2)
    boundary_terrain_ids = jax_vmap(_get_tid)(boundary_beyond)         # (B,)

    terrain_ids = jnp.concatenate([obs_terrain_ids, boundary_terrain_ids])  # (2B,)
    return hit_points, terrain_ids


# ---------------------------------------------------------------------------
# Robust training variants V1–V4
# ---------------------------------------------------------------------------
# V1/V2: speed + noise; boundary terrain IDs zeroed to Grass (1) so the policy
#         sees no terrain-lane information — pure navigation without terrain awareness.
# V3/V4: speed + noise; boundary terrain IDs populated from the sim geometry —
#         terrain-aware rewards (PREF_VECTOR, TERRAIN_REWARD) enabled.
# V2/V4 add 500ms (15-step) action delay on top of their respective base.

class LidarTargetV1(LidarTarget):
    """V1: real-world speed (1.5 m/s) + lidar noise + state noise. No terrain boundary info."""

    MAX_SPEED_MS: float = 1.5
    SCALE_2D_3D: float = 11.0
    SIM_MAX_VEL: float = 1.5 / 11.0   # ≈ 0.1364 sim units/s

    # Terrain-boundary rewards disabled — boundary terrain IDs are all Grass
    PREF_VECTOR_REWARD_COEFF = 0.0
    TERRAIN_REWARD_COEFF = 0.0

    PARAMS = {
        **LidarTarget.PARAMS,
        "lidar_noise_std": 0.05 / 11.0,   # 5 cm real → ~0.0045 sim units
        "pos_noise_std":   0.03 / 11.0,   # 3 cm position → ~0.0027 sim units
        "vel_noise_std":   0.074 / 11.0,  # 0.074 m/s velocity (measured) → ~0.0067 sim units
    }

    def reset(self, key, **kwargs):
        graph = super().reset(key, **kwargs)
        env_state = graph.env_states
        # Initialize obs_agent with (n_agents, 4) so lax.scan carry shape is consistent with step()
        new_env_state = env_state._replace(
            obs_agent=jnp.zeros((self.num_agents, 4), dtype=jnp.float32)
        )
        return graph._replace(env_states=new_env_state)

    def state_lim(self, state=None):
        v = self.SIM_MAX_VEL
        return jnp.array([0., 0., -v, -v]), jnp.array([self.area_size, self.area_size, v, v])

    def agent_step_euler(self, agent_states, action):
        vel = action * self.SIM_MAX_VEL
        next_pos = agent_states[:, :2] + vel * self.dt
        return self.clip_state(jnp.concatenate([next_pos, vel], axis=1))

    def get_semantic_lidar_data(self, states, obstacles, bridge_center, bridge_gap_width,
                                bridge_wall_thickness, bridge_theta, terrain_config,
                                bridge_length=1.0, bridge_bend_angle=0.0):
        """Compute lidar but null out boundary terrain IDs (set all to Grass=1)."""
        from dgppo.env.lidar_env.base import LidarEnv
        lidar_data, terrain_ids = LidarEnv.get_semantic_lidar_data(
            self, states, obstacles, bridge_center, bridge_gap_width,
            bridge_wall_thickness, bridge_theta, terrain_config,
            bridge_length=bridge_length, bridge_bend_angle=bridge_bend_angle,
        )
        n_rays = self._params["n_rays"]
        # First n_rays = obstacle hits (keep as-is); last n_rays = boundary hits (null to Grass)
        terrain_ids_no_bnd = terrain_ids.at[:, n_rays:].set(1)
        return lidar_data, terrain_ids_no_bnd

    def _obs_noise(self, agent_states, lidar_data, key):
        k1, k2, k3 = jr.split(key, 3)
        pos_noise = jr.normal(k1, (self.num_agents, 2)) * self.params["pos_noise_std"]
        vel_noise = jr.normal(k2, (self.num_agents, 2)) * self.params["vel_noise_std"]
        obs_agent = agent_states + jnp.concatenate([pos_noise, vel_noise], axis=1)
        noisy_lidar = (
            lidar_data + jr.normal(k3, lidar_data.shape) * self.params["lidar_noise_std"]
            if lidar_data is not None
            else lidar_data
        )
        return obs_agent, noisy_lidar


class LidarTargetV2(LidarTargetV1):
    """V2: V1 features + 500ms (15-step) action delay."""

    def reset(self, key, **kwargs):
        graph = super().reset(key, **kwargs)
        env_state = graph.env_states
        new_env_state = env_state._replace(
            action_buffer=jnp.zeros(
                (DELAY_STEPS, self.num_agents, self.action_dim), dtype=jnp.float32
            )
        )
        return self.get_graph(new_env_state, None)

    def _get_executed_action(self, action, env_state):
        buf = env_state.action_buffer                                    # (DELAY_STEPS, n_agents, 2)
        delayed_action = buf[0]                                          # (n_agents, 2) — oldest
        new_buf = jnp.concatenate([buf[1:], action[None]], axis=0)      # FIFO shift
        return delayed_action, env_state._replace(action_buffer=new_buf)


class LidarTargetV3(LidarTargetV1):
    """V3: V1 features + real terrain boundary hits + preference vector reward."""

    PREF_VECTOR_REWARD_COEFF = 0.2

    def get_semantic_lidar_data(self, states, obstacles, bridge_center, bridge_gap_width,
                                bridge_wall_thickness, bridge_theta, terrain_config,
                                bridge_length=1.0, bridge_bend_angle=0.0):
        """Use real terrain-boundary lidar (skips V1's all-grass override)."""
        lidar_data, terrain_ids = jax_vmap(
            ft.partial(
                _get_semantic_lidar_single_real,
                obstacles=obstacles,
                bridge_center=bridge_center,
                bridge_gap_width=bridge_gap_width,
                bridge_wall_thickness=bridge_wall_thickness,
                bridge_theta=bridge_theta,
                terrain_config=terrain_config,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
                bridge_length=bridge_length,
                bridge_bend_angle=bridge_bend_angle,
            )
        )(states[:, :2])
        return lidar_data, terrain_ids


class LidarTargetV4(LidarTargetV2):
    """V4: V2 features (speed + noise + delay) + real terrain boundary hits + preference vector reward."""

    PREF_VECTOR_REWARD_COEFF = 0.2

    def get_semantic_lidar_data(self, states, obstacles, bridge_center, bridge_gap_width,
                                bridge_wall_thickness, bridge_theta, terrain_config,
                                bridge_length=1.0, bridge_bend_angle=0.0):
        """Use real terrain-boundary lidar (skips V1's all-grass override)."""
        lidar_data, terrain_ids = jax_vmap(
            ft.partial(
                _get_semantic_lidar_single_real,
                obstacles=obstacles,
                bridge_center=bridge_center,
                bridge_gap_width=bridge_gap_width,
                bridge_wall_thickness=bridge_wall_thickness,
                bridge_theta=bridge_theta,
                terrain_config=terrain_config,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
                bridge_length=bridge_length,
                bridge_bend_angle=bridge_bend_angle,
            )
        )(states[:, :2])
        return lidar_data, terrain_ids
