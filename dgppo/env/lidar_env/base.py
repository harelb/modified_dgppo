import pathlib
import jax.numpy as jnp
import jax.random as jr
import numpy as np # Keep if needed for other non-JAX ops, but avoid for JAX arrays
import functools as ft
import jax
import jax.debug as jd
import ipdb

from typing import NamedTuple, Tuple, Optional, List, Dict # Added List, Dict for type hints
from abc import ABC, abstractmethod

from jaxtyping import Float

from ...trainer.data import Rollout
from ...utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ...utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState, PRNGKey
from ...utils.utils import merge01, jax_vmap
from ..base import MultiAgentEnv
from dgppo.env.obstacle import Obstacle, Rectangle
from dgppo.env.plot import render_lidar # Ensure BehaviorAssociator is imported from plot
from dgppo.env.utils import get_lidar, get_node_goal_rng, get_ray_alphas
from dgppo.env.bridge_behavior_associator import BehaviorBridge

class LidarEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle

    bearing: Float[Array, "n_agent"]
    current_cluster_oh: Float[Array, "n_agent n_cluster"]
    start_cluster_oh: Float[Array, "n_agent n_cluster"]
    next_cluster_oh: Float[Array, "n_agent n_cluster"]
    next_cluster_bonus_awarded: jnp.ndarray

    # Terrain one-hot: [Road, Grass, Sidewalk]
    current_terrain_oh: Float[Array, "n_agent 3"]
    # Semantic lidar — full 2*n_rays hits per agent, stored flat
    lidar_hit_terrain_ids: jnp.ndarray   # (n_agents * 2*n_rays,)  terrain ID per hit
    lidar_hit_positions:   jnp.ndarray   # (n_agents * 2*n_rays, 2) world position per hit

    bridge_center: Float[Array, "2"]
    bridge_length: float
    bridge_gap_width: float
    bridge_wall_thickness: float
    bridge_theta: float         # Stored in radians; angle of segment 1 (entry segment)
    bridge_bend_angle: float    # Angular offset of segment 2 relative to segment 1 (0 = straight)
    terrain_config: jnp.ndarray  # scalar int: 1 or 2, sampled randomly per episode

    # Noise/delay variant fields (defaults = no-op for base LidarTarget)
    key: jnp.ndarray = jnp.zeros(2, dtype=jnp.uint32)       # PRNGKey carried through steps
    obs_agent: jnp.ndarray = jnp.zeros((0, 4))               # noisy agent obs; (0,4)=use true state
    action_buffer: jnp.ndarray = jnp.zeros((0, 2))           # delay buffer; (0,2)=no delay

    # Observation delay buffers (lag variants)
    obs_agent_buffer: jnp.ndarray = jnp.zeros((0, 0, 4))    # (buf_size, n_agents, 4)
    lidar_buffer: jnp.ndarray = jnp.zeros((0, 0, 0, 2))     # (buf_size, n_agents, 2*n_rays, 2)

    # Reward delay buffer (lag variants) — kept for backward compat but no longer used
    reward_buffer: jnp.ndarray = jnp.zeros(0)                       # (act_delay,); empty=no shift

    # Terrain randomization flag (DRT) — sampled once per episode at reset
    use_real_terrain: jnp.ndarray = jnp.ones((), dtype=jnp.int32)  # 1=real, 0=hardcoded

    # Bug 3/4 fix: additional observation delay buffers for lag variants
    lidar_terrain_id_buffer: jnp.ndarray = jnp.zeros((0, 0, 0), dtype=jnp.int32)  # (buf_size, n_agents, 2*n_rays)
    bearing_buffer: jnp.ndarray = jnp.zeros((0, 0))                                # (buf_size, n_agents)
    terrain_oh_buffer: jnp.ndarray = jnp.zeros((0, 0, 0))                          # (buf_size, n_agents, 3)
    cluster_oh_buffer: jnp.ndarray = jnp.zeros((0, 0, 0))                          # (buf_size, n_agents, n_cluster)

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]
    
    @property
    def n_cluster(self) -> int:
        return 4


LidarEnvGraphsTuple = GraphsTuple[State, LidarEnvState]

# Terrain IDs: Road=0, Grass=1, Sidewalk=2
# terrain_config controls detection geometry (see get_terrain_id).
TERRAIN_NAMES = ["Road", "Grass", "Sidewalk"]
TERRAIN_CONFIG = 1  # Switch between 1 and 2 here

# Target terrain: the zone the robot should be in.
# Semantic lidar reports distance to the nearest boundary OF this zone per ray:
#   - if in target  → distance to exit edge (avoid crossing it)
#   - if not in target → distance to entry edge (approach it)
TARGET_TERRAIN_ID = 2  # Sidewalk


def _terrain_from_perp(
    perp: jnp.ndarray,               # scalar, absolute perpendicular distance from corridor axis
    bridge_gap_width: jnp.ndarray,
    bridge_wall_thickness: jnp.ndarray,
    terrain_config: jnp.ndarray,
) -> jnp.ndarray:
    """Terrain ID from absolute perpendicular distance to a corridor centerline."""
    half_gap  = bridge_gap_width / 2.0
    full_half = half_gap + bridge_wall_thickness
    sidewalk_border = bridge_gap_width * 0.2
    road_half = half_gap - sidewalk_border
    is_road     = perp <= road_half
    is_sidewalk = (perp > road_half) & (perp <= half_gap)
    tid_c1 = jnp.where(perp <= full_half, 2, 1)
    tid_c2 = jnp.where(is_road, 0, jnp.where(is_sidewalk, 2, 1))
    return jnp.where(terrain_config == 1, tid_c1, tid_c2)


def get_terrain_id(
    agent_pos: jnp.ndarray,             # (2,)
    bridge_center: jnp.ndarray,          # (2,) — bend/junction point for bent bridge
    bridge_gap_width: jnp.ndarray,       # scalar
    bridge_wall_thickness: jnp.ndarray,  # scalar
    bridge_theta: jnp.ndarray,           # scalar, radians — angle of segment 1 (entry)
    terrain_config: jnp.ndarray,         # scalar int: 1 or 2
    bridge_length: jnp.ndarray = 1.0,   # total bridge length (each segment = half)
    bridge_bend_angle: jnp.ndarray = 0.0,  # angular offset of segment 2 (0 = straight)
) -> jnp.ndarray:
    """
    Geometry-based terrain detection.

    Straight bridge (bridge_bend_angle == 0):
      Uses infinite perpendicular strip from bridge_center (original behaviour).

    Bent bridge (bridge_bend_angle != 0):
      Two segments meeting at bridge_center (the bend point).
        Segment 1: entry → bridge_center, direction bridge_theta
        Segment 2: bridge_center → exit, direction bridge_theta + bridge_bend_angle
      Longitudinally clamped per segment (10 % overlap at junction).
      Terrain = segment with smaller perp distance when in both; Grass if in neither.

    terrain_config = 1 : full band = Sidewalk
    terrain_config = 2 : Road | Sidewalk | Grass
    """
    # ── Straight-bridge terrain (original infinite-strip logic) ──────────────
    dx_s = agent_pos[0] - bridge_center[0]
    dy_s = agent_pos[1] - bridge_center[1]
    cos_t = jnp.cos(bridge_theta)
    sin_t = jnp.sin(bridge_theta)
    perp_straight = jnp.abs(-sin_t * dx_s + cos_t * dy_s)
    tid_straight = _terrain_from_perp(perp_straight, bridge_gap_width, bridge_wall_thickness, terrain_config)

    # ── Bent-bridge terrain (two longitudinally-clamped segments) ────────────
    seg_quarter = bridge_length / 4.0   # distance from bridge_center to each segment center

    # Segment 1 (entry → bridge_center)
    seg1_dir = jnp.array([cos_t, sin_t])
    seg1_center = bridge_center - seg_quarter * seg1_dir
    dp1 = agent_pos - seg1_center
    perp1 = jnp.abs(-sin_t * dp1[0] + cos_t * dp1[1])

    # Segment 2 (bridge_center → exit)
    theta2 = bridge_theta + bridge_bend_angle
    cos_t2 = jnp.cos(theta2)
    sin_t2 = jnp.sin(theta2)
    seg2_dir = jnp.array([cos_t2, sin_t2])
    seg2_center = bridge_center + seg_quarter * seg2_dir
    dp2 = agent_pos - seg2_center
    perp2 = jnp.abs(-sin_t2 * dp2[0] + cos_t2 * dp2[1])

    tid1 = _terrain_from_perp(perp1, bridge_gap_width, bridge_wall_thickness, terrain_config)
    tid2 = _terrain_from_perp(perp2, bridge_gap_width, bridge_wall_thickness, terrain_config)

    # Split plane at the kink (bridge_center) using the bisector of the two segment directions.
    # Points on the entry side use seg1 terrain (extends beyond entry end);
    # points on the exit side use seg2 terrain (extends beyond exit end).
    bisector = seg1_dir + seg2_dir   # proportional to the angular bisector
    dp_kink = agent_pos - bridge_center
    on_seg2_side = jnp.dot(dp_kink, bisector) >= 0
    tid_bent = jnp.where(on_seg2_side, tid2, tid1)

    return jnp.where(jnp.abs(bridge_bend_angle) < 1e-6, tid_straight, tid_bent)


def _get_semantic_lidar_single(
    agent_pos: jnp.ndarray,              # (2,)
    obstacles,                           # Obstacle (stacked)
    bridge_center: jnp.ndarray,          # (2,) — bend/junction point
    bridge_gap_width: jnp.ndarray,
    bridge_wall_thickness: jnp.ndarray,
    bridge_theta: jnp.ndarray,           # segment-1 angle (radians)
    terrain_config: jnp.ndarray,
    num_beams: int,
    sense_range: float,
    bridge_length: jnp.ndarray = 1.0,   # total bridge length
    bridge_bend_angle: jnp.ndarray = 0.0,  # segment-2 offset angle (0 = straight)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-ray semantic lidar for one agent.

    For each of the B rays, returns TWO hit points (2B total):
      [0 : B]  — obstacle hit (or sensor-range endpoint if no obstacle)
      [B : 2B] — nearest boundary of TARGET_TERRAIN_ID zone along that ray

    For a bent bridge the terrain boundaries from BOTH segments are considered;
    the nearest valid one is returned.  For a straight bridge (bend_angle=0)
    both segments are collinear so the result is identical to the original.

    Returns:
        hit_points:  (2*num_beams, 2)   positions in world frame
        terrain_ids: (2*num_beams,)     terrain at each hit point (0=Road,1=Grass,2=Sidewalk)
    """
    thetas = jnp.linspace(-jnp.pi, jnp.pi - 2 * jnp.pi / num_beams, num_beams)
    dirs   = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)  # (B, 2)
    starts = jnp.tile(agent_pos[None, :], (num_beams, 1))             # (B, 2)
    ends   = starts + dirs * sense_range                               # (B, 2)

    # ── Obstacle hit alphas (one per ray) ──────────────────────────────────
    alphas_obs = get_ray_alphas(starts, ends, obstacles)               # (B,)

    # ── Build hit points ────────────────────────────────────────────────────
    # Boundary hits hard-coded to zeros; terrain IDs all Grass=1.
    # Matches execution: spot_dgppo_ros_node.py bnd_hits = np.zeros((n_rays, 2))
    # and all_terrain_ids = np.ones(2*n_rays).
    obs_hits      = agent_pos[None, :] + alphas_obs[:, None] * sense_range * dirs  # (B, 2)
    boundary_hits = agent_pos[None, :] + sense_range * dirs                          # (B, 2)
    hit_points    = jnp.concatenate([obs_hits, boundary_hits])                      # (2B, 2)

    terrain_ids = jnp.ones(2 * num_beams, dtype=jnp.int32)  # all Grass, matches execution

    return hit_points, terrain_ids


def _topk_hits_per_agent(
    hits_2ray: jnp.ndarray,   # (2*n_rays, 2) — obstacle hits [0:n_rays] + boundary hits [n_rays:]
    tids_2ray: jnp.ndarray,   # (2*n_rays,)
    agent_pos: jnp.ndarray,   # (2,)
    n_rays: int,
    top_k: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Select top-k closest obstacle hits and top-k closest boundary hits for one agent."""
    obs_hits = hits_2ray[:n_rays];  bnd_hits = hits_2ray[n_rays:]
    obs_tids = tids_2ray[:n_rays];  bnd_tids = tids_2ray[n_rays:]

    obs_dists = jnp.linalg.norm(obs_hits - agent_pos[None, :], axis=-1)
    bnd_dists = jnp.linalg.norm(bnd_hits - agent_pos[None, :], axis=-1)

    _, obs_idx = jax.lax.top_k(-obs_dists, top_k)
    _, bnd_idx = jax.lax.top_k(-bnd_dists, top_k)

    return (
        jnp.concatenate([obs_hits[obs_idx], bnd_hits[bnd_idx]]),  # (2*top_k, 2)
        jnp.concatenate([obs_tids[obs_idx], bnd_tids[bnd_idx]]),  # (2*top_k,)
    )


def create_bent_bridge(
    bridge_center: jnp.ndarray,  # (2,) — bend/junction point
    bridge_length: float,        # total length; each segment = bridge_length/2
    bridge_gap_width: float,
    bridge_wall_thickness: float,
    bridge_theta: float,         # segment-1 angle, radians
    bridge_bend_angle: float,    # segment-2 offset from segment-1 (0 = straight)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create 4 rectangular obstacles for a (possibly bent) bridge.

    Wall layout (indices in returned arrays):
      0 — segment-1, left  wall  (+perp side)
      1 — segment-1, right wall  (-perp side)
      2 — segment-2, left  wall  (+perp side)
      3 — segment-2, right wall  (-perp side)

    For bridge_bend_angle == 0 the two segments are collinear and the result
    is a straight bridge split into two equal halves (backward-compatible
    with geometry-based collision / lidar, though the obstacle count is 4
    instead of 2 compared to the old create_single_bridge).
    """
    seg_len     = bridge_length / 2.0        # each segment's length
    seg_quarter = bridge_length / 4.0        # bridge_center → segment center offset
    half_dist   = (bridge_gap_width / 2.0) + (bridge_wall_thickness / 2.0)

    # ── Segment 1 (entry → bridge_center) ────────────────────────────────
    cos1, sin1 = jnp.cos(bridge_theta), jnp.sin(bridge_theta)
    seg1_center = bridge_center - seg_quarter * jnp.array([cos1, sin1])
    perp1 = jnp.array([-sin1, cos1]) * half_dist   # perpendicular offset
    wall0_center = seg1_center + perp1              # left wall
    wall1_center = seg1_center - perp1              # right wall

    # ── Segment 2 (bridge_center → exit) ─────────────────────────────────
    theta2 = bridge_theta + bridge_bend_angle
    cos2, sin2 = jnp.cos(theta2), jnp.sin(theta2)
    seg2_center = bridge_center + seg_quarter * jnp.array([cos2, sin2])
    perp2 = jnp.array([-sin2, cos2]) * half_dist
    wall2_center = seg2_center + perp2              # left wall
    wall3_center = seg2_center - perp2              # right wall

    # ── Outer-arc extension: close the gap at the outer kink corner ───────
    # The outer wall pair (right for CCW bend, left for CW) doesn't meet at
    # the kink.  Extend each outer wall past bridge_center by exactly the
    # longitudinal projection of the gap onto that segment's axis.
    outer_half_dist = bridge_gap_width / 2.0 + bridge_wall_thickness
    outer_ext = outer_half_dist * jnp.abs(jnp.sin(bridge_bend_angle))
    zero2 = jnp.zeros(2)
    ccw   = bridge_bend_angle > 0   # CCW: outer=right (wall1,wall3); CW: outer=left (wall0,wall2)

    # Shift each outer seg-1 wall center toward the kink (+seg1_dir) by half the extension
    seg1_step = (outer_ext / 2.0) * jnp.array([cos1, sin1])
    wall0_center = wall0_center + jnp.where(~ccw, seg1_step, zero2)   # left  extends when CW
    wall1_center = wall1_center + jnp.where( ccw, seg1_step, zero2)   # right extends when CCW

    # Shift each outer seg-2 wall center back toward the kink (-seg2_dir) by half the extension
    seg2_step = (outer_ext / 2.0) * jnp.array([cos2, sin2])
    wall2_center = wall2_center - jnp.where(~ccw, seg2_step, zero2)
    wall3_center = wall3_center - jnp.where( ccw, seg2_step, zero2)

    ext0 = jnp.where(~ccw, outer_ext, 0.0)
    ext1 = jnp.where( ccw, outer_ext, 0.0)
    ext2 = jnp.where(~ccw, outer_ext, 0.0)
    ext3 = jnp.where( ccw, outer_ext, 0.0)

    obs_pos   = jnp.stack([wall0_center, wall1_center, wall2_center, wall3_center])       # (4, 2)
    obs_len_x = jnp.array([seg_len + ext0, seg_len + ext1, seg_len + ext2, seg_len + ext3])  # (4,)
    obs_len_y = jnp.full((4,), bridge_wall_thickness)                                         # (4,)
    obs_theta = jnp.array([bridge_theta, bridge_theta, theta2, theta2])                       # (4,)

    return obs_pos, obs_len_x, obs_len_y, obs_theta

class LidarEnv(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        "num_bridges": 1, # Default to 1 bridge - Important for this scenario
        "bridge_length_range": [0.5, 1.0],
        "bridge_gap_width_range": [0.2, 0.4],
        "bridge_wall_thickness_range": [0.05, 0.1],
        "open_space_goal_distance": 0.2, # Added parameter for open space goal placement
        "goal_offset_from_centroid": 0.0 
    }
    
    ALL_POSSIBLE_REGION_NAMES = [
        "open_space",
        "approach_bridge_0",
        "on_bridge_0",
        "exit_bridge_0"
        
    ]
    
    # CLUSTER_MAP now defines the canonical integer IDs for region types
    CLUSTER_MAP: Dict[str, int] = {
        "open_space": 0,
        "approach_bridge_0": 1,
        "on_bridge_0": 2, 
        "exit_bridge_0":3
        
        #"around_bridge_0": 4, # You may also have "start" or "cross_bridge_gap" - ensure consistency
    }
    
    # Define curriculum transitions as a Python list of string tuples (for human readability/initial setup)
    _CURRICULUM_TRANSITIONS_STR = [ # Renamed to avoid confusion with the JAX array
        #("open_space", "approach_bridge_0"),
        ("approach_bridge_0", "on_bridge_0"), # Using "cross_bridge_gap" from plot.py
        ("on_bridge_0", "exit_bridge_0"),
        ("exit_bridge_0", "open_space"),
    ]

    # Chain: when an agent reaches its next_cluster, advance to this cluster.
    # Sequence: approach(1) → on(2) → exit(3) → open_space(0), then stop (open_space loops to itself).
    _CHAIN_NEXT_STR: Dict[str, str] = {
        "approach_bridge_0": "on_bridge_0",
        "on_bridge_0":       "exit_bridge_0",
        "exit_bridge_0":     "open_space",
        "open_space":        "open_space",   # terminal: stay in open_space
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 128,
            dt: float = 0.03,
            params: dict = None,
            n_cluster: int = 4
    ):
        area_size = LidarEnv.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarEnv, self).__init__(num_agents, area_size, max_step, dt, params)
        self.create_obstacles = jax_vmap(Rectangle.create)
        self.num_goals = self._num_agents
        self.n_cluster = n_cluster
        
        if self.n_cluster < len(set(self.CLUSTER_MAP.values())):
            print(f"Warning: n_cluster ({self.n_cluster}) is less than the number of unique cluster IDs in CLUSTER_MAP ({len(set(self.CLUSTER_MAP.values()))}).")

        # Map integer IDs back to *base* string names (e.g., 0 -> "open_space", not "open_space_0")
        self._id_to_curriculum_prefix_map: Dict[int, str] = {v: k for k, v in self.CLUSTER_MAP.items()}
        
        allowed_id_transitions_list: List[Tuple[int, int]] = []
        for start_prefix, end_prefix in self._CURRICULUM_TRANSITIONS_STR:
            # Use .get() with a default value (e.g., -1) to handle prefixes not in CLUSTER_MAP gracefully,
            # though they should ideally all be defined.
            start_id = self.CLUSTER_MAP.get(start_prefix, -1) 
            end_id = self.CLUSTER_MAP.get(end_prefix, -1)
            if start_id != -1 and end_id != -1: # Only add if both IDs are valid
                allowed_id_transitions_list.append((start_id, end_id))
            else:
                print(f"Warning: Curriculum transition ({start_prefix}, {end_prefix}) contains unknown cluster prefixes.")
        
        # Convert to JAX array. If the list is empty, create an empty (0, 2) array.
        if allowed_id_transitions_list:
            self.CURRICULUM_TRANSITIONS_INT_IDS = jnp.array(allowed_id_transitions_list, dtype=jnp.int32)
        else:
            self.CURRICULUM_TRANSITIONS_INT_IDS = jnp.empty((0, 2), dtype=jnp.int32)

        # Build per-cluster chain lookup: CHAIN_NEXT_IDS[i] = next cluster id after reaching cluster i.
        chain_list = [
            self.CLUSTER_MAP.get(self._CHAIN_NEXT_STR.get(name, name), i)
            for i, name in enumerate(self.ALL_POSSIBLE_REGION_NAMES)
        ]
        self.CHAIN_NEXT_IDS = jnp.array(chain_list, dtype=jnp.int32)  # (n_cluster,)
        
    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy
    
    @property
    def bearing_dim(self) -> int:
        return 1 # Bearing is a single float
    
    @property
    def cluster_oh_dim(self) -> int:
        return self.n_cluster # One-hot encoding dimension

    @property
    def terrain_oh_dim(self) -> int:
        return 3  # Road=0, Grass=1, Sidewalk=2

    @property
    def agent_extra_dim(self) -> int:
        """Extra features appended to agent nodes (e.g., flattened action buffer). Base: 0."""
        return 0

    def _agent_extra_features(self, state: "LidarEnvState") -> jnp.ndarray:
        """Returns (n_agents, agent_extra_dim) extra features for agent nodes. Base: empty."""
        return jnp.zeros((self.num_agents, 0), dtype=jnp.float32)

    @property
    def node_dim(self) -> int:
        # +4 indicators: is_obs | is_terrain_boundary | is_goal | is_agent
        return self.state_dim + self.bearing_dim + 3 * self.cluster_oh_dim + self.terrain_oh_dim + 4 + self.agent_extra_dim
    
    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_vel, vy_vel

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def reset(
        self,
        key: PRNGKey,
        current_clusters: Optional[Array] = None,
        start_clusters: Optional[Array] = None,
        next_clusters: Optional[Array] = None,
        custom_obstacles: Optional[Tuple[Array, Array, Array, Array]] = None,
        transition_index: Optional[int] = None # This is now ONLY for starting a specific episode
    ) -> GraphsTuple:
        
        # jd.print("new episode")
    
        if current_clusters is None:
            current_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)
        if start_clusters is None:
            start_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)
        if next_clusters is None:
            next_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)

        all_obs_pos_list: List[jnp.ndarray] = []
        all_obs_len_x_list: List[jnp.ndarray] = []
        all_obs_len_y_list: List[jnp.ndarray] = []
        all_obs_theta_list: List[jnp.ndarray] = []

        bridge_center_env_state: Float[Array, "2"] = jnp.array([0.0, 0.0])
        bridge_length_env_state: Float[Array, ""] = jnp.array(0.0)
        bridge_gap_width_env_state: Float[Array, ""] = jnp.array(0.0)
        bridge_wall_thickness_env_state: Float[Array, ""] = jnp.array(0.0)
        bridge_theta_env_state: Float[Array, ""] = jnp.array(0.0)
        bridge_bend_angle_env_state: Float[Array, ""] = jnp.array(0.0)
        terrain_config_env_state: jnp.ndarray = jnp.array(1, dtype=jnp.int32)
        initial_bonus_awarded = jnp.zeros(self.num_agents, dtype=jnp.bool_)

        num_bridges = 1 

        # jd.print("transition index {}", transition_index)
        if num_bridges > 0:
            if transition_index is None:
                # Original logic: pick a random transition
                transition_idx_key, key = jr.split(key)
                num_transitions = self.CURRICULUM_TRANSITIONS_INT_IDS.shape[0]
                transition_index = jr.randint(transition_idx_key, (), 0, num_transitions)
            
            start_region_id_tracer, next_region_id_tracer = self.CURRICULUM_TRANSITIONS_INT_IDS[transition_index]
            # jd.print("current index {}", start_region_id_tracer)
            # jd.print("next index {}", next_region_id_tracer)
            
        if custom_obstacles is not None:
            obs_pos, obs_len_x, obs_len_y, obs_theta = custom_obstacles
            all_obs_pos_list.append(obs_pos)
            all_obs_len_x_list.append(obs_len_x)
            all_obs_len_y_list.append(obs_len_y)
            all_obs_theta_list.append(obs_theta)
        else:
            n_rng_obs = self._params["n_obs"]
            assert n_rng_obs >= 0

            if n_rng_obs > 0:
                obstacle_key, key = jr.split(key, 2)
                obs_pos_orig = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
                length_key, key = jr.split(key, 2)
                obs_len_orig = jr.uniform(
                    length_key,
                    (n_rng_obs, 2),
                    minval=self._params["obs_len_range"][0],
                    maxval=self._params["obs_len_range"][1],
                )
                theta_key, key = jr.split(key, 2)
                obs_theta_orig = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * jnp.pi)

                all_obs_pos_list.append(obs_pos_orig)
                all_obs_len_x_list.append(obs_len_orig[:, 0])
                all_obs_len_y_list.append(obs_len_orig[:, 1])
                all_obs_theta_list.append(obs_theta_orig) 
                
            if num_bridges > 0:
                if num_bridges != 1:
                    print("Warning: Only 1 bridge is currently supported for direct parameter storage.")

                bridge_rand_key, key = jr.split(key)
                center_key, length_key, gap_key, thickness_key, theta_key, config_key, bend_key = jr.split(bridge_rand_key, 7)

                bridge_length = jr.uniform(length_key, (),
                                            minval=self._params.get("bridge_length_range", (0.5, 1.0))[0],
                                            maxval=self._params.get("bridge_length_range", (0.5, 1.0))[1])
                bridge_gap_width = jr.uniform(gap_key, (),
                                                minval=self._params.get("bridge_gap_width_range", (0.2, 0.4))[0],
                                                maxval=self._params.get("bridge_gap_width_range", (0.2, 0.4))[1])
                bridge_wall_thickness = jr.uniform(thickness_key, (),
                                                    minval=self._params.get("bridge_wall_thickness_range", (0.03, 0.1))[0],
                                                    maxval=self._params.get("bridge_wall_thickness_range", (0.03, 0.1))[1])
                bridge_theta = jr.uniform(theta_key, (), minval=0, maxval=2 * jnp.pi)
                terrain_config_episode = jr.randint(config_key, (), minval=1, maxval=3)  # 1 or 2
                _bend_range = self._params.get("bridge_bend_angle_range", (0.0, 0.0))
                bridge_bend_angle = jr.uniform(bend_key, (),
                                               minval=_bend_range[0],
                                               maxval=_bend_range[1])

                effective_len = bridge_length
                effective_width = bridge_gap_width + 2 * bridge_wall_thickness
                
                cos_abs = jnp.abs(jnp.cos(bridge_theta))
                sin_abs = jnp.abs(jnp.sin(bridge_theta))

                max_x_extent = 0.5 * (effective_len * cos_abs + effective_width * sin_abs)
                max_y_extent = 0.5 * (effective_len * sin_abs + effective_width * cos_abs)

                min_center_x = max_x_extent
                max_center_x = self.area_size - max_x_extent
                min_center_y = max_y_extent
                max_center_y = self.area_size - max_y_extent

                min_center_x = jnp.maximum(0.0, min_center_x)
                max_center_x = jnp.maximum(min_center_x, max_center_x) # Ensure max is not less than min

                min_center_y = jnp.maximum(0.0, min_center_y)
                max_center_y = jnp.maximum(min_center_y, max_center_y) # Ensure max is not less than min
                bridge_center = jr.uniform(center_key, (2,),
                                            minval=jnp.array([min_center_x, min_center_y]),
                                            maxval=jnp.array([max_center_x, max_center_y]))
                
                # bridge_center = jnp.array([0.75, 0.2])
                # bridge_length = 1
                # bridge_gap_width = 0.3
                # bridge_wall_thickness = 0.06
                # bridge_theta = 0

                bridge_obs_pos, bridge_obs_len_x, bridge_obs_len_y, bridge_obs_theta = \
                    create_bent_bridge(
                        bridge_center,
                        bridge_length,
                        bridge_gap_width,
                        bridge_wall_thickness,
                        bridge_theta,
                        bridge_bend_angle,
                    )
                
                all_obs_pos_list.append(bridge_obs_pos)
                all_obs_len_x_list.append(bridge_obs_len_x)
                all_obs_len_y_list.append(bridge_obs_len_y)
                all_obs_theta_list.append(bridge_obs_theta)

                bridge_center_env_state = bridge_center
                bridge_length_env_state = bridge_length
                bridge_gap_width_env_state = bridge_gap_width
                bridge_wall_thickness_env_state = bridge_wall_thickness
                bridge_theta_env_state = bridge_theta
                bridge_bend_angle_env_state = bridge_bend_angle
                terrain_config_env_state = terrain_config_episode

        if all_obs_pos_list:
            combined_obs_pos = jnp.concatenate(all_obs_pos_list, axis=0)
            combined_obs_len_x = jnp.concatenate(all_obs_len_x_list, axis=0)
            combined_obs_len_y = jnp.concatenate(all_obs_len_y_list, axis=0)
            combined_obs_theta = jnp.concatenate(all_obs_theta_list, axis=0)
            obstacles = self.create_obstacles(
                combined_obs_pos,
                combined_obs_len_x,
                combined_obs_len_y,
                combined_obs_theta
            )
        else:
            obstacles = None

        if num_bridges > 0:

            # Build 4-wall params for BehaviorBridge (matches create_bent_bridge layout).
            # Wall 0/1 = segment-1 (left/right), Wall 2/3 = segment-2 (left/right).
            _half_dist = (bridge_gap_width_env_state / 2.0) + (bridge_wall_thickness_env_state / 2.0)
            _seg_len   = bridge_length_env_state / 2.0
            _seg_q     = bridge_length_env_state / 4.0
            _theta1    = bridge_theta_env_state
            _theta2    = bridge_theta_env_state + bridge_bend_angle_env_state
            _cos1, _sin1 = jnp.cos(_theta1), jnp.sin(_theta1)
            _cos2, _sin2 = jnp.cos(_theta2), jnp.sin(_theta2)
            _seg1_c = bridge_center_env_state - _seg_q * jnp.array([_cos1, _sin1])
            _seg2_c = bridge_center_env_state + _seg_q * jnp.array([_cos2, _sin2])
            _perp1  = jnp.array([-_sin1, _cos1]) * _half_dist
            _perp2  = jnp.array([-_sin2, _cos2]) * _half_dist
            _w0 = _seg1_c + _perp1;  _w1 = _seg1_c - _perp1
            _w2 = _seg2_c + _perp2;  _w3 = _seg2_c - _perp2

            bridges_for_associator = [
                (_w0[0], _w0[1], _seg_len, bridge_wall_thickness_env_state, jnp.degrees(_theta1)),
                (_w1[0], _w1[1], _seg_len, bridge_wall_thickness_env_state, jnp.degrees(_theta1)),
                (_w2[0], _w2[1], _seg_len, bridge_wall_thickness_env_state, jnp.degrees(_theta2)),
                (_w3[0], _w3[1], _seg_len, bridge_wall_thickness_env_state, jnp.degrees(_theta2)),
            ]

            associator = BehaviorBridge(
                bridges=bridges_for_associator,
                all_region_names=self.ALL_POSSIBLE_REGION_NAMES
            )

            agent_states_list = []
            goal_states_list = []
            bearings_list = []
            current_clusters_oh_list = []
            start_clusters_oh_list = []
            next_clusters_oh_list = []

            open_space_id = self.CLUSTER_MAP["open_space"]
            exit_bridge_id = self.CLUSTER_MAP["exit_bridge_0"]
            
            bridge_dir_vec = jnp.array([jnp.cos(bridge_theta_env_state), jnp.sin(bridge_theta_env_state)])
            theta2_env = bridge_theta_env_state + bridge_bend_angle_env_state
            seg2_dir_vec = jnp.array([jnp.cos(theta2_env), jnp.sin(theta2_env)])

            bridge_end1 = bridge_center_env_state - (bridge_length_env_state / 2) * bridge_dir_vec
            bridge_end2 = bridge_center_env_state + (bridge_length_env_state / 2) * seg2_dir_vec
            
            exit_bridge_centroid = associator.get_region_centroid(exit_bridge_id)
            
            exit_direction_vector = seg2_dir_vec
            
            open_space_goal_distance = 0.4 #self.params["open_space_goal_distance"]
            
            open_space_goal_pos = exit_bridge_centroid + exit_direction_vector * open_space_goal_distance

            for i in range(self.num_agents):
                key_agent_pos, key_goal_pos, key_rot, key_loop = jr.split(key, 4)
                key = key_loop

                initial_pos_candidate = associator.get_region_centroid(start_region_id_tracer)
                
                initial_pos = jnp.where(jnp.isnan(initial_pos_candidate).any(), 
                                        jnp.zeros(2), # Default if NaN
                                        initial_pos_candidate)
                
                initial_pos = initial_pos + jr.normal(key_agent_pos, (2,)) * 0.05
                initial_pos = jnp.clip(initial_pos, 0, self.area_size)

                goal_pos_candidate = associator.get_region_centroid(next_region_id_tracer)
                
                is_open_space_goal = (next_region_id_tracer == self.CLUSTER_MAP["open_space"])

                goal_pos = jnp.where(
                    is_open_space_goal,
                    open_space_goal_pos,
                    goal_pos_candidate
                )

                goal_pos = jnp.where(jnp.isnan(goal_pos).any(), 
                                    jnp.array([self.area_size / 2, self.area_size / 2]), # Fallback to center, not origin
                                    goal_pos)
                
                goal_pos = goal_pos + jr.normal(key_goal_pos, (2,)) * 0.05
    
                rot_angle = jr.uniform(key_rot, (), minval=-jnp.pi/4, maxval=jnp.pi/4)

                cos_rot = jnp.cos(rot_angle)
                sin_rot = jnp.sin(rot_angle)
                
                rotation_matrix = jnp.array([
                    [cos_rot, -sin_rot],
                    [sin_rot, cos_rot]
                ])

                goal_pos_relative = goal_pos - bridge_center_env_state
                rotated_goal_pos_relative = jnp.dot(rotation_matrix, goal_pos_relative)
                rotated_goal_pos = bridge_center_env_state + rotated_goal_pos_relative
                
                goal_pos = rotated_goal_pos

                bearing = jnp.arctan2(goal_pos[1] - initial_pos[1], goal_pos[0] - initial_pos[0]) #- jnp.pi/4

                start_cluster_idx = start_region_id_tracer 
                current_cluster_idx = jnp.squeeze(associator.get_current_behavior(initial_pos))
                next_cluster_idx = next_region_id_tracer
                
                current_cluster_oh = jax.nn.one_hot(current_cluster_idx, self.n_cluster)
                start_cluster_oh = jax.nn.one_hot(start_cluster_idx, self.n_cluster)
                next_cluster_oh = jax.nn.one_hot(next_cluster_idx, self.n_cluster)

                agent_states_list.append(jnp.array([initial_pos[0], initial_pos[1], 0.0, 0.0]))
                goal_states_list.append(jnp.array([goal_pos[0], goal_pos[1], 0.0, 0.0]))
                bearings_list.append(bearing)
                current_clusters_oh_list.append(current_cluster_oh)
                start_clusters_oh_list.append(start_cluster_oh)
                next_clusters_oh_list.append(next_cluster_oh)
            
            states = jnp.stack(agent_states_list)
            goals = jnp.stack(goal_states_list)
            bearing = jnp.stack(bearings_list)
            current_clusters = jnp.stack(current_clusters_oh_list)
            start_clusters = jnp.stack(start_clusters_oh_list)
            next_clusters = jnp.stack(next_clusters_oh_list)

            # --- Terrain one-hot at reset (geometry-based, independent of clusters) ---
            initial_terrain_ids = jax_vmap(
                ft.partial(
                    get_terrain_id,
                    bridge_center=bridge_center_env_state,
                    bridge_gap_width=bridge_gap_width_env_state,
                    bridge_wall_thickness=bridge_wall_thickness_env_state,
                    bridge_theta=bridge_theta_env_state,
                    terrain_config=terrain_config_env_state,
                    bridge_length=bridge_length_env_state,
                    bridge_bend_angle=bridge_bend_angle_env_state,
                )
            )(states[:, :2])  # (n_agents,)
            initial_terrain_oh = jax.nn.one_hot(initial_terrain_ids, self.terrain_oh_dim)  # (n_agents, 3)
            # jd.print("[TERRAIN RESET DEBUG] terrain_ids (0=Road,1=Grass,2=Sidewalk): {}", initial_terrain_ids)
            # jd.print("[TERRAIN RESET DEBUG] terrain_oh: {}", initial_terrain_oh)

        else: # Fallback if no bridges are generated or BehaviorAssociator not used
            states, goals = get_node_goal_rng(
                key, self.area_size, 2, self.num_agents, 2.2 * self._params["car_radius"], obstacles)

            states = jnp.concatenate(
                [states, jnp.zeros((self.num_agents, self.state_dim - states.shape[1]), dtype=states.dtype)], axis=1)
            goals = jnp.concatenate(
                [goals, jnp.zeros((self.num_goals, self.state_dim - goals.shape[1]), dtype=goals.dtype)], axis=1)

            bearing = jnp.zeros((self.num_agents,), dtype=jnp.float32)
            current_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)
            next_clusters = jnp.zeros((self.num_agents, self.n_cluster), dtype=jnp.float32)
            # Default: all Grass when no bridge present
            initial_terrain_oh = jnp.zeros((self.num_agents, self.terrain_oh_dim), dtype=jnp.float32)
            initial_terrain_oh = initial_terrain_oh.at[:, 1].set(1.0)  # Grass


        assert states.shape == (self.num_agents, self.state_dim)
        assert goals.shape == (self.num_goals, self.state_dim)

        lidar_data, lidar_terrain_ids = self.get_semantic_lidar_data(
            states, obstacles,
            bridge_center_env_state, bridge_gap_width_env_state,
            bridge_wall_thickness_env_state, bridge_theta_env_state,
            terrain_config_env_state,
            bridge_length=bridge_length_env_state,
            bridge_bend_angle=bridge_bend_angle_env_state,
            use_real_terrain=jnp.array(1, dtype=jnp.int32),
        )
        n_full = self.num_agents * 2 * self._params["n_rays"]
        if lidar_data is not None:
            lidar_hit_terrain_ids_flat = merge01(lidar_terrain_ids)        # (n_full,)
            lidar_hit_positions_flat   = merge01(lidar_data)               # (n_full, 2)
        else:
            lidar_hit_terrain_ids_flat = jnp.ones(n_full, dtype=jnp.int32)
            lidar_hit_positions_flat   = jnp.zeros((n_full, 2), dtype=jnp.float32)

        step_key, key = jr.split(key)
        env_states = LidarEnvState(
            agent=states,
            goal=goals,
            obstacle=obstacles,
            bearing=bearing,
            current_cluster_oh=current_clusters,
            start_cluster_oh=start_clusters,
            next_cluster_oh=next_clusters,
            next_cluster_bonus_awarded=initial_bonus_awarded,
            current_terrain_oh=initial_terrain_oh,
            lidar_hit_terrain_ids=lidar_hit_terrain_ids_flat,
            lidar_hit_positions=lidar_hit_positions_flat,
            bridge_center=bridge_center_env_state,
            bridge_length=bridge_length_env_state,
            bridge_gap_width=bridge_gap_width_env_state,
            bridge_wall_thickness=bridge_wall_thickness_env_state,
            bridge_theta=bridge_theta_env_state,
            bridge_bend_angle=bridge_bend_angle_env_state,
            terrain_config=terrain_config_env_state,
            key=step_key,
        )

        return self.get_graph(env_states, lidar_data)
    
    # def print_get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
    #     lidar_data = None
    #     if self.params["n_obs"] > 0:
    #         get_lidar_vmap = jax_vmap(
    #             ft.partial(
    #                 get_lidar,
    #                 obstacles=obstacles,
    #                 num_beams=self._params["n_rays"],
    #                 sense_range=self._params["comm_radius"],
    #                 max_returns=32
    #             )
    #         )
    #         lidar_data = get_lidar_vmap(states[:, :2])
    #         assert lidar_data.shape == (self.num_agents, 32, 2)
    #     return lidar_data

    def get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
        lidar_data = None
        if self.params["n_obs"] > 0:
            get_lidar_vmap = jax_vmap(
                ft.partial(
                    get_lidar,
                    obstacles=obstacles,
                    num_beams=self._params["n_rays"],
                    sense_range=self._params["comm_radius"],
                    max_returns=self._params["top_k_rays"],
                )
            )
            lidar_data = get_lidar_vmap(states[:, :2])
            assert lidar_data.shape == (self.num_agents, self._params["top_k_rays"], 2)
        return lidar_data

    def get_semantic_lidar_data(
        self,
        states: State,
        obstacles,
        bridge_center: jnp.ndarray,
        bridge_gap_width: jnp.ndarray,
        bridge_wall_thickness: jnp.ndarray,
        bridge_theta: jnp.ndarray,
        terrain_config: jnp.ndarray,
        bridge_length: jnp.ndarray = 1.0,
        bridge_bend_angle: jnp.ndarray = 0.0,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Per-ray semantic lidar.
        Returns:
            lidar_data:   (n_agents, 2*n_rays, 2)  hit positions
            terrain_ids:  (n_agents, 2*n_rays)     terrain ID per hit
        First n_rays entries per agent = obstacle hits; last n_rays = target-terrain boundary hits.
        """
        lidar_data, terrain_ids = jax_vmap(
            ft.partial(
                _get_semantic_lidar_single,
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
        return lidar_data, terrain_ids  # (n_agents, 2*n_rays, 2), (n_agents, 2*n_rays)

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        # Velocity control: action is directly the velocity command, scaled to state limits
        vel = action * 0.5  # action in [-1,1] -> velocity in [-0.5, 0.5]
        next_pos = agent_states[:, :2] + vel * self.dt
        n_state_agent_new = jnp.concatenate([next_pos, vel], axis=1)
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: LidarEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarEnvGraphsTuple, Reward, Cost, Done, Info]:
        agent_base_states = graph.env_states.agent # Get the underlying agent state from env_states
        goals = graph.env_states.goal
        obstacles = graph.env_states.obstacle
        
        bridge_center = graph.env_states.bridge_center
        bridge_length = graph.env_states.bridge_length
        bridge_gap_width = graph.env_states.bridge_gap_width
        bridge_wall_thickness = graph.env_states.bridge_wall_thickness
        bridge_theta = graph.env_states.bridge_theta
        bridge_bend_angle = graph.env_states.bridge_bend_angle
        terrain_config = graph.env_states.terrain_config

        cos_theta = jnp.cos(bridge_theta)
        sin_theta = jnp.sin(bridge_theta)

        # Rebuild 4-wall params matching create_bent_bridge layout.
        _half_dist_s = (bridge_gap_width / 2.0) + (bridge_wall_thickness / 2.0)
        _seg_len_s   = bridge_length / 2.0
        _seg_q_s     = bridge_length / 4.0
        _theta1_s    = bridge_theta
        _theta2_s    = bridge_theta + bridge_bend_angle
        _cos1_s, _sin1_s = jnp.cos(_theta1_s), jnp.sin(_theta1_s)
        _cos2_s, _sin2_s = jnp.cos(_theta2_s), jnp.sin(_theta2_s)
        _seg1_cs = bridge_center - _seg_q_s * jnp.array([_cos1_s, _sin1_s])
        _seg2_cs = bridge_center + _seg_q_s * jnp.array([_cos2_s, _sin2_s])
        _perp1_s = jnp.array([-_sin1_s, _cos1_s]) * _half_dist_s
        _perp2_s = jnp.array([-_sin2_s, _cos2_s]) * _half_dist_s
        _sw0 = _seg1_cs + _perp1_s;  _sw1 = _seg1_cs - _perp1_s
        _sw2 = _seg2_cs + _perp2_s;  _sw3 = _seg2_cs - _perp2_s

        bridge_walls_params = [
            (_sw0[0], _sw0[1], _seg_len_s, bridge_wall_thickness, jnp.degrees(_theta1_s)),
            (_sw1[0], _sw1[1], _seg_len_s, bridge_wall_thickness, jnp.degrees(_theta1_s)),
            (_sw2[0], _sw2[1], _seg_len_s, bridge_wall_thickness, jnp.degrees(_theta2_s)),
            (_sw3[0], _sw3[1], _seg_len_s, bridge_wall_thickness, jnp.degrees(_theta2_s)),
        ]

        bearing = graph.env_states.bearing
        from dgppo.env.bridge_behavior_associator import BehaviorBridge
        associator = BehaviorBridge(
            bridges=bridge_walls_params,
            all_region_names=self.ALL_POSSIBLE_REGION_NAMES
        )
        current_id = jax_vmap(associator.get_current_behavior)(agent_base_states[:, :2])
        current_cluster_oh = jax.nn.one_hot(current_id, self.n_cluster)
        start_cluster_oh = graph.env_states.start_cluster_oh
        next_cluster_oh = graph.env_states.next_cluster_oh

        # calculate next states
        action = self.clip_action(action)
        executed_action, env_state_updated = self._get_executed_action(action, graph.env_states)
        next_agent_base_states = self.agent_step_euler(agent_base_states, executed_action)

        # --- Terrain one-hot: geometry-based detection on next positions ---
        next_terrain_ids = jax_vmap(
            ft.partial(
                get_terrain_id,
                bridge_center=bridge_center,
                bridge_gap_width=bridge_gap_width,
                bridge_wall_thickness=bridge_wall_thickness,
                bridge_theta=bridge_theta,
                terrain_config=terrain_config,
                bridge_length=bridge_length,
                bridge_bend_angle=bridge_bend_angle,
            )
        )(next_agent_base_states[:, :2])  # (n_agents,)
        next_terrain_oh = jax.nn.one_hot(next_terrain_ids, self.terrain_oh_dim)  # (n_agents, 3)

        # jd.print("[TERRAIN STEP DEBUG] agent positions: {}", next_agent_base_states[:, :2])
        # jd.print("[TERRAIN STEP DEBUG] terrain_ids (0=Road,1=Grass,2=Sidewalk): {}", next_terrain_ids)
        # jd.print("[TERRAIN STEP DEBUG] terrain_oh: {}", next_terrain_oh)

        reward, bonus_awarded_updated = self.get_reward(graph, executed_action)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()

        chained_next_cluster_oh  = next_cluster_oh
        chained_start_cluster_oh = start_cluster_oh

        lidar_data_next, lidar_terrain_ids_next = self.get_semantic_lidar_data(
            next_agent_base_states, obstacles,
            bridge_center, bridge_gap_width,
            bridge_wall_thickness, bridge_theta, terrain_config,
            bridge_length=bridge_length,
            bridge_bend_angle=bridge_bend_angle,
            use_real_terrain=env_state_updated.use_real_terrain,
        )
        lidar_hit_positions_next = merge01(lidar_data_next)

        step_key, k_noise = jr.split(env_state_updated.key)
        obs_agent, noisy_lidar = self._obs_noise(next_agent_base_states, lidar_data_next, k_noise)
        (obs_agent, noisy_lidar, lidar_terrain_ids_next,
         bearing, next_terrain_oh, current_cluster_oh,
         env_state_updated) = self._apply_obs_delay(
            obs_agent, noisy_lidar, lidar_terrain_ids_next,
            bearing, next_terrain_oh, current_cluster_oh, env_state_updated
        )

        next_env_state = LidarEnvState(
            next_agent_base_states,
            goals,
            obstacles,
            bearing,
            current_cluster_oh,
            chained_start_cluster_oh,
            chained_next_cluster_oh,
            bonus_awarded_updated,
            next_terrain_oh,
            merge01(lidar_terrain_ids_next),
            lidar_hit_positions_next,
            bridge_center,
            bridge_length,
            bridge_gap_width,
            bridge_wall_thickness,
            bridge_theta,
            bridge_bend_angle,
            terrain_config,
            key=step_key,
            obs_agent=obs_agent,
            action_buffer=env_state_updated.action_buffer,
            obs_agent_buffer=env_state_updated.obs_agent_buffer,
            lidar_buffer=env_state_updated.lidar_buffer,
            reward_buffer=env_state_updated.reward_buffer,
            use_real_terrain=env_state_updated.use_real_terrain,
            lidar_terrain_id_buffer=env_state_updated.lidar_terrain_id_buffer,
            bearing_buffer=env_state_updated.bearing_buffer,
            terrain_oh_buffer=env_state_updated.terrain_oh_buffer,
            cluster_oh_buffer=env_state_updated.cluster_oh_buffer,
        )

        info = {}
        done = jnp.array(False)

        return self.get_graph(next_env_state, noisy_lidar), reward, cost, done, info

    @abstractmethod
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        pass

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)

        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # Hard collision cost: only obstacle hit nodes (first top_k per agent in graph).
        # Boundary hit nodes are NOT obstacles — they are soft terrain navigation signals.
        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            top_k = self._params["top_k_rays"]
            all_hit_nodes = graph.type_states(type_idx=2, n_type=2 * top_k * self.num_agents)[:, :2]
            # Reshape to (n_agents, 2*top_k, 2); first top_k per agent = obstacle hits
            all_hit_nodes = jnp.reshape(all_hit_nodes, (self.num_agents, 2 * top_k, 2))
            obs_pos = all_hit_nodes[:, :top_k, :]                      # (n_agents, top_k, 2)
            dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agents, top_k)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agents,)

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, -1.0, 1.0)

        return cost

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        from dgppo.env.plot import render_lidar 

        first_env_state = rollout.graph.env_states
        bridge_params_for_render = {
            "bridge_center": first_env_state.bridge_center,
            "bridge_length": first_env_state.bridge_length,
            "bridge_gap_width": first_env_state.bridge_gap_width,
            "bridge_wall_thickness": first_env_state.bridge_wall_thickness,
            "bridge_theta": first_env_state.bridge_theta,
        }

        render_lidar(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=2 * self.params["top_k_rays"] if self.params["n_obs"] > 0 or self.params["num_bridges"] > 0 else 0,
            r=self.params["car_radius"],
            obs_r=0.0, #
            cost_components=self.cost_components,
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            n_goal=self.num_goals,
            dpi=dpi,
            **bridge_params_for_render, 
            **kwargs )

    @abstractmethod
    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        pass
    
    def get_graph(self, state: LidarEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        n_rays = self._params["n_rays"]
        top_k  = self._params["top_k_rays"]
        # Graph uses top-k per type; rewards use full data from env_states
        n_hits  = 2 * top_k * self.num_agents
        n_nodes = self.num_agents + self.num_goals + n_hits

        # ── Select top-k obstacle and boundary hits per agent ──────────────
        if lidar_data is not None:
            # lidar_data: (n_agents, 2*n_rays, 2)
            tid_per_agent = jnp.reshape(
                state.lidar_hit_terrain_ids[:2 * n_rays * self.num_agents],
                (self.num_agents, 2 * n_rays),
            )
            ld_topk, tids_topk = jax_vmap(
                ft.partial(_topk_hits_per_agent, n_rays=n_rays, top_k=top_k)
            )(lidar_data, tid_per_agent, state.agent[:, :2])
            # ld_topk: (n_agents, 2*top_k, 2)  |  tids_topk: (n_agents, 2*top_k)
            lidar_data_g = merge01(ld_topk)    # (n_agents * 2*top_k, 2)
            tids_g       = merge01(tids_topk)  # (n_agents * 2*top_k,)
        else:
            lidar_data_g = jnp.zeros((n_hits, 2), dtype=jnp.float32)
            tids_g       = jnp.ones(n_hits, dtype=jnp.int32)  # default Grass

        # Node feature layout:
        #   [:state_dim]       — kinematic state
        #   [state_dim]        — bearing
        #   [+n_cluster*3]     — current/start/next cluster OH
        #   [+terrain_oh_dim]  — terrain OH
        #   [IND+0] is_obs | [IND+1] is_terrain_boundary | [IND+2] is_goal | [IND+3] is_agent
        IND = self.state_dim + self.bearing_dim + 3 * self.n_cluster + self.terrain_oh_dim
        node_feats = jnp.zeros((n_nodes, self.node_dim), dtype=jnp.float32)

        # Agent nodes — use noisy obs when available (obs_agent shape (0,4) means use true state)
        agent_for_obs = state.obs_agent if state.obs_agent.shape[0] > 0 else state.agent
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(agent_for_obs)
        node_feats = node_feats.at[:self.num_agents, self.state_dim].set(state.bearing)
        node_feats = node_feats.at[:self.num_agents, self.state_dim+self.bearing_dim:self.state_dim+self.bearing_dim+self.n_cluster].set(state.current_cluster_oh)
        node_feats = node_feats.at[:self.num_agents, self.state_dim+self.bearing_dim+self.n_cluster:self.state_dim+self.bearing_dim+2*self.n_cluster].set(state.start_cluster_oh)
        node_feats = node_feats.at[:self.num_agents, self.state_dim+self.bearing_dim+2*self.n_cluster:self.state_dim+self.bearing_dim+3*self.n_cluster].set(state.next_cluster_oh)
        node_feats = node_feats.at[:self.num_agents, self.state_dim+self.bearing_dim+3*self.n_cluster:IND].set(state.current_terrain_oh)
        node_feats = node_feats.at[:self.num_agents, IND+3].set(1.0)  # is_agent

        # Goal nodes
        node_feats = node_feats.at[self.num_agents:self.num_agents+self.num_goals, :self.state_dim].set(state.goal)
        node_feats = node_feats.at[self.num_agents:self.num_agents+self.num_goals, IND+2].set(1.0)  # is_goal

        # Lidar hit nodes (top-k per type per agent)
        n_obs_g = top_k * self.num_agents
        n_bnd_g = top_k * self.num_agents
        obs_s   = self.num_agents + self.num_goals
        bnd_s   = obs_s + n_obs_g

        node_feats = node_feats.at[obs_s:obs_s+n_obs_g, :2].set(lidar_data_g[:n_obs_g])
        node_feats = node_feats.at[bnd_s:bnd_s+n_bnd_g, :2].set(lidar_data_g[n_obs_g:])

        hit_terrain_oh = jax.nn.one_hot(tids_g[:n_hits], self.terrain_oh_dim)
        node_feats = node_feats.at[obs_s:obs_s+n_obs_g, self.state_dim+self.bearing_dim+3*self.n_cluster:IND].set(hit_terrain_oh[:n_obs_g])
        node_feats = node_feats.at[bnd_s:bnd_s+n_bnd_g, self.state_dim+self.bearing_dim+3*self.n_cluster:IND].set(hit_terrain_oh[n_obs_g:])
        node_feats = node_feats.at[obs_s:obs_s+n_obs_g, IND+0].set(1.0)  # is_obs
        node_feats = node_feats.at[bnd_s:bnd_s+n_bnd_g, IND+1].set(1.0)  # is_terrain_boundary

        # Node types
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(LidarEnv.AGENT)
        node_type = node_type.at[self.num_agents:self.num_agents+self.num_goals].set(LidarEnv.GOAL)
        if n_hits > 0:
            node_type = node_type.at[self.num_agents+self.num_goals:].set(LidarEnv.OBS)

        edge_blks = self.edge_blocks(state, lidar_data_g)

        # Raw states (top-k positions for graph, consistent with node_feats)
        raw_states = jnp.concatenate([state.agent, state.goal], axis=0)
        lidar_states_padded = jnp.concatenate(
            [lidar_data_g, jnp.zeros((n_hits, self.state_dim - 2), dtype=lidar_data_g.dtype)],
            axis=1,
        )
        raw_states = jnp.concatenate([raw_states, lidar_states_padded], axis=0)

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blks,
            env_states=state,
            states=raw_states,
        ).to_padded()
    
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -0.5, -0.5])
        upper_lim = jnp.array([self.area_size, self.area_size, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def _get_executed_action(
        self, action: Action, env_state: "LidarEnvState"
    ) -> Tuple[Action, "LidarEnvState"]:
        """Returns (executed_action, updated_env_state). Base: action executes immediately."""
        return action, env_state

    def _apply_obs_delay(
        self,
        obs_agent: jnp.ndarray,
        noisy_lidar: Optional[jnp.ndarray],
        terrain_ids: Optional[jnp.ndarray],
        bearing: jnp.ndarray,
        terrain_oh: jnp.ndarray,
        cluster_oh: jnp.ndarray,
        env_state: "LidarEnvState",
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, "LidarEnvState"]:
        """Returns (obs_agent, lidar, terrain_ids, bearing, terrain_oh, cluster_oh, env_state). Base: no delay."""
        return obs_agent, noisy_lidar, terrain_ids, bearing, terrain_oh, cluster_oh, env_state

    def _obs_noise(
        self,
        agent_states: jnp.ndarray,
        lidar_data: Optional[jnp.ndarray],
        key: PRNGKey,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Returns (obs_agent, noisy_lidar_data). Base: no noise; obs_agent shape (0,4)."""
        return jnp.zeros((0, 4), dtype=jnp.float32), lidar_data
