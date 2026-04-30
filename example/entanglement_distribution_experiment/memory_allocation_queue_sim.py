"""Memory-allocation validation for two elementary links and one swap.

The analytic queue model is compared against a SeQUeNCe-backed simulation.
The SeQUeNCe path uses a small controller for asymmetric elementary-link memory
banks such as 1:5.  The controller starts the standard single-heralded
generation protocols on selected memories and uses the shared SeQUeNCe-style
resource-manager request/response path for swapping.
"""

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from math import exp, isnan
import os
from statistics import mean, stdev
from time import monotonic
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.resource_management.memory_manager import MemoryInfo

from example.entanglement_distribution_experiment.eg_single_heralded_helpers import (
    ElementaryLinkConfig,
    EntanglementLifetimePolicy,
    ExtendedBSMNode,
    ExtendedQuantumRouter,
    FiberSpec,
    SwapCalibration,
    _classical_channel_delay_ps,
    _configure_generated_lifetime_controls,
    _connect_classical_fully,
    _install_generation_pair,
    _make_quantum_channel,
    _shortest_path_distances,
    _use_pair_level_single_heralded,
    start_rule_verified_bell_swap,
    PS_PER_S,
)
from example.queueing_model import queue_utils as q_model


@dataclass
class AllocationQueueConfig:
    """Inputs for the memory-allocation queue study."""

    left_rate_hz: float = 300.0
    right_rate_hz: float = 720.0
    left_distance_km: float = 32.0
    right_distance_km: float = 18.0
    left_raw_fidelity: float = 0.9
    right_raw_fidelity: float = 0.9
    swap_degradation: float = 1.0
    coherence_time_s: float = 0.1
    swap_success_prob: float = 1.0
    classical_speed_km_s: float = 2e5
    classical_message_overhead_s: float = 10e-6
    optic_depol_left: float = 0.03
    optic_depol_right: float = 0.02
    optical_coherence_length_km: float = 250.0
    memory_decoherence_rates_hz: tuple[float, float, float] = (5.0, 10.0, 10.0)
    memory_noise_type: str = "dephasing"

    @property
    def left_signal_time_s(self) -> float:
        return self.left_distance_km / self.classical_speed_km_s + self.classical_message_overhead_s

    @property
    def right_signal_time_s(self) -> float:
        return self.right_distance_km / self.classical_speed_km_s + self.classical_message_overhead_s

    @property
    def swap_time_s(self) -> float:
        # One swap-control round trip to the slowest end node. Each signal time
        # already includes the fixed SeQUeNCe classical-channel overhead.
        return 2 * max(self.left_signal_time_s, self.right_signal_time_s)

    def left_link_config(self) -> ElementaryLinkConfig:
        return ElementaryLinkConfig(
            label="left",
            distance_m=self.left_distance_km * 1000,
            coherence_time_s=self.coherence_time_s,
            memory_decoherence_errors=[1 / 3, 1 / 3, 1 / 3],
            raw_fidelity=self.left_raw_fidelity,
            fiber_spec=FiberSpec(wavelength_m=1550e-9, quantum_wavelength_nm=1550.0),
            swap_node_name="r2",
        )

    def right_link_config(self) -> ElementaryLinkConfig:
        return ElementaryLinkConfig(
            label="right",
            distance_m=self.right_distance_km * 1000,
            coherence_time_s=self.coherence_time_s,
            memory_decoherence_errors=[1 / 3, 1 / 3, 1 / 3],
            raw_fidelity=self.right_raw_fidelity,
            fiber_spec=FiberSpec(wavelength_m=1550e-9, quantum_wavelength_nm=1550.0),
            swap_node_name="r2",
        )

    def swap_config(self) -> SwapCalibration:
        return SwapCalibration(success_prob=self.swap_success_prob, degradation=self.swap_degradation)


def ttl_for_required_fidelity(
    required_fidelity: float,
    *,
    raw_fidelity: float,
    coherence_time_s: float,
) -> float:
    """Return deterministic TTL that keeps pair fidelity above the target.

    This uses the same qualitative policy as the attached study: a pair is
    allowed to wait until its fidelity falls to the requested minimum.  If the
    requested fidelity is higher than raw fidelity, the TTL is zero.
    """

    if required_fidelity <= 0:
        return coherence_time_s
    if required_fidelity >= raw_fidelity:
        return 0.0
    return max(0.0, min(coherence_time_s, -coherence_time_s * np.log(required_fidelity / raw_fidelity)))


def reference_model_terms(config: AllocationQueueConfig) -> dict[str, Any]:
    """Return constants matching the shared queueing-model notebook.

    The optical-channel conversion is delegated to the canonical queueing-model code.
    This wrapper only maps our config object into the model variables.
    """

    t1 = config.left_signal_time_s
    t2 = config.right_signal_time_s
    tau1 = 1 * t1
    tau2 = 1 * t2
    optic_depha_left = 1 - np.exp(-config.left_distance_km / 2 / config.optical_coherence_length_km)
    optic_depha_right = 1 - np.exp(-config.right_distance_km / 2 / config.optical_coherence_length_km)
    f_raw1, raw_errors1 = q_model.channel2Pauli(config.optic_depol_left, optic_depha_left)
    f_raw2, raw_errors2 = q_model.channel2Pauli(config.optic_depol_right, optic_depha_right)
    v1, w1 = 1 - config.optic_depol_left, 1 - optic_depha_left
    v2, w2 = 1 - config.optic_depol_right, 1 - optic_depha_right
    v_swp = v1 * v2 * config.swap_degradation
    w_swp = w1 * w2
    gam0, gam1, gam2 = config.memory_decoherence_rates_hz
    gam01, gam12 = gam0 + gam1, gam1 + gam2
    tswp = config.swap_time_s
    a0 = gam0 * (t1 + tswp) + gam2 * (t2 + tswp) + gam01 * tau1 + gam12 * tau2
    a0 += gam1 * 2 * tswp

    if config.memory_noise_type == "dephasing":
        waittime2fidelity = lambda x: (1 + v_swp * (1 + 2 * w_swp * np.exp(-(a0 if x is None else a0 + x)))) / 4
        threshold = lambda f_req: -np.log(((4 * f_req - 1) / v_swp - 1) / (2 * w_swp))
    elif config.memory_noise_type == "depolarizing":
        waittime2fidelity = lambda x: (1 + v_swp * np.exp(-(a0 if x is None else a0 + x)) * (1 + 2 * w_swp)) / 4
        threshold = lambda f_req: -np.log(((4 * f_req - 1) / v_swp / (1 + 2 * w_swp)))
    else:
        raise ValueError('memory_noise_type must be "dephasing" or "depolarizing"')

    return {
        "T1": t1,
        "T2": t2,
        "tau1": tau1,
        "tau2": tau2,
        "Tswp": tswp,
        "f_raw1": f_raw1,
        "f_raw2": f_raw2,
        "raw_errors1": raw_errors1,
        "raw_errors2": raw_errors2,
        "gam01": gam01,
        "gam12": gam12,
        "A0": a0,
        "F_max": float(waittime2fidelity(0)),
        "waittime2fidelity": waittime2fidelity,
        "threshold": threshold,
        "f1": {"raw_epr": f_raw1, "raw_errors": raw_errors1, "dephase_rates": (gam0, gam1), "decohere_rates": (gam0, gam1)},
        "f2": {"raw_epr": f_raw2, "raw_errors": raw_errors2, "dephase_rates": (gam1, gam2), "decohere_rates": (gam1, gam2)},
    }


def reference_ttls_for_required_fidelity(
    config: AllocationQueueConfig,
    required_fidelity: float,
) -> tuple[float, float]:
    terms = reference_model_terms(config)
    a_thr = terms["threshold"](required_fidelity)
    return max(0.0, float((a_thr - terms["A0"]) / terms["gam01"])), max(0.0, float((a_thr - terms["A0"]) / terms["gam12"]))


def pair_fidelity_after_age(raw_fidelity: float, age_s: float, coherence_time_s: float) -> float:
    """Simple exponential pair-fidelity decay used by the allocation simulator."""

    if age_s <= 0:
        return raw_fidelity
    if coherence_time_s <= 0:
        return 0.0
    return float(raw_fidelity * np.exp(-age_s / coherence_time_s))



def reference_swap_2q(q1: tuple[Any, ...], q2: tuple[Any, ...], tswp: float = 0.0, method: str = "full") -> tuple[float, float]:
    """Call the canonical queueing-model implementation from example/queueing_model/queue_utils.py."""

    q_model_method = "simple" if method == "simple" else "exact"
    rate, wait_time = q_model.swap_2Q(q1, q2, Tswp=tswp, method=q_model_method, capacity="multi")
    return float(rate), float(wait_time)


def allocation_model_rate(
    left_capacity: int,
    right_capacity: int,
    *,
    multiplexing: bool,
    left_rate_hz: float,
    right_rate_hz: float,
    left_ttl_s: float,
    right_ttl_s: float,
    swap_time_s: float,
    left_reset_s: float = 0.0,
    right_reset_s: float = 0.0,
) -> float:
    """Reference notebook model rate for an allocation."""

    lam1 = [left_rate_hz * (left_capacity - k) if left_capacity - k > 0 else 0 for k in range(left_capacity)] if multiplexing else left_rate_hz
    lam2 = [right_rate_hz * (right_capacity - k) if right_capacity - k > 0 else 0 for k in range(right_capacity)] if multiplexing else right_rate_hz
    q1 = (left_capacity, lam1, 0.0, left_ttl_s, left_reset_s, {"decohere_rates": (1.0, 1.0)})
    q2 = (right_capacity, lam2, 0.0, right_ttl_s, right_reset_s, {"decohere_rates": (1.0, 1.0)})
    rate, _ = reference_swap_2q(q1, q2, tswp=swap_time_s, method="full")
    return rate


def _truncated_exponential_mean(rate_hz: float, window_s: float) -> float:
    if rate_hz <= 0 or window_s <= 0:
        return 0.0
    p = 1 - exp(-rate_hz * window_s)
    if p <= 0:
        return 0.0
    return (1 / rate_hz) - (window_s * exp(-rate_hz * window_s) / p)


def allocation_model_fidelity(
    config: AllocationQueueConfig,
    left_capacity: int,
    right_capacity: int,
    *,
    multiplexing: bool,
    left_ttl_s: float,
    right_ttl_s: float,
) -> float:
    terms = reference_model_terms(config)
    lam1 = [config.left_rate_hz * (left_capacity - k) if left_capacity - k > 0 else 0 for k in range(left_capacity)] if multiplexing else config.left_rate_hz
    lam2 = [config.right_rate_hz * (right_capacity - k) if right_capacity - k > 0 else 0 for k in range(right_capacity)] if multiplexing else config.right_rate_hz
    q1 = (left_capacity, lam1, terms["tau1"], left_ttl_s, terms["T1"], terms["f1"])
    q2 = (right_capacity, lam2, terms["tau2"], right_ttl_s, terms["T2"], terms["f2"])
    _, wait_time = reference_swap_2q(q1, q2, tswp=terms["Tswp"], method="full")
    return float(terms["waittime2fidelity"](wait_time))


def _memory_has_generation_protocol(memory) -> bool:
    """Return True when a memory is occupied by elementary generation, not swap control."""

    for observer in getattr(memory, "_observers", []):
        if "SingleHeralded" in observer.__class__.__name__:
            return True
    return False


class _MemoryAllocationController:
    def __init__(
        self,
        *,
        timeline: Timeline,
        r1: ExtendedQuantumRouter,
        r2: ExtendedQuantumRouter,
        r3: ExtendedQuantumRouter,
        m12: ExtendedBSMNode,
        m23: ExtendedBSMNode,
        left_cfg: ElementaryLinkConfig,
        right_cfg: ElementaryLinkConfig,
        swap_cfg: SwapCalibration,
        left_capacity: int,
        right_capacity: int,
        multiplexing: bool,
        target_fidelity: float,
        stop_time_ps: int,
        control_period_ps: int,
        swap_selection_policy: str,
    ) -> None:
        self.timeline = timeline
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.m12 = m12
        self.m23 = m23
        self.left_cfg = left_cfg
        self.right_cfg = right_cfg
        self.swap_cfg = swap_cfg
        self.left_capacity = left_capacity
        self.right_capacity = right_capacity
        self.multiplexing = multiplexing
        self.target_fidelity = target_fidelity
        self.stop_time_ps = stop_time_ps
        self.control_period_ps = control_period_ps
        if swap_selection_policy not in {"random", "fifo"}:
            raise ValueError('swap_selection_policy must be "random" or "fifo"')
        self.swap_selection_policy = swap_selection_policy
        self.protocol_counter = 0
        self.swap_counter = 0
        self.delivery_times_s: List[float] = []
        self.delivery_fidelities: List[float] = []
        self.accepted_delivery_times_s: List[float] = []
        self.accepted_delivery_fidelities: List[float] = []
        self.left_waits_s: List[float] = []
        self.right_waits_s: List[float] = []
        self.left_arrival_times_s: List[float] = []
        self.right_arrival_times_s: List[float] = []
        self.state_trace: list[dict[str, Any]] = []
        self._seen_elementary_arrivals: set[tuple[str, int]] = set()
        self.memory_owner = {}
        for router in (r1, r2, r3):
            for memory in router.get_components_by_type("MemoryArray")[0]:
                self.memory_owner[memory.name] = router
        self._in_reaction = False
        self._reaction_scheduled = False
        self._install_resource_update_hooks()

    def start(self) -> None:
        self._request_reaction()

    def _schedule_tick(self, time_ps: int) -> None:
        if time_ps <= self.stop_time_ps:
            self.timeline.schedule(Event(time_ps, Process(self, "tick", [])))

    def tick(self) -> None:
        self._record_elementary_arrivals()
        self._consume_delivered_pairs()
        if self.timeline.now() >= self.stop_time_ps:
            return
        self._start_ready_swaps()
        self._start_generation_attempts()
        self._schedule_tick(self.timeline.now() + self.control_period_ps)

    def _install_resource_update_hooks(self) -> None:
        for router in (self.r1, self.r2, self.r3):
            original_update = router.resource_manager.update

            def wrapped_update(protocol, memory, state, *, _original_update=original_update):
                result = _original_update(protocol, memory, state)
                self._request_reaction()
                return result

            router.resource_manager.update = wrapped_update

    def _request_reaction(self) -> None:
        if self._reaction_scheduled or self.timeline.now() >= self.stop_time_ps:
            return
        self._reaction_scheduled = True
        self.timeline.schedule(Event(self.timeline.now(), Process(self, "_react_to_state_change", [])))

    def _react_to_state_change(self) -> None:
        self._reaction_scheduled = False
        if self._in_reaction or self.timeline.now() >= self.stop_time_ps:
            return
        self._in_reaction = True
        try:
            self._record_state_trace()
            self._record_elementary_arrivals()
            self._consume_delivered_pairs()
            self._start_ready_swaps()
            self._start_generation_attempts()
            self._record_state_trace()
        finally:
            self._in_reaction = False

    def _info(self, router, memory):
        return router.resource_manager.memory_manager.get_info_by_memory(memory)

    def _state(self, router, memory) -> str:
        return self._info(router, memory).state

    def _count_states(self, memories) -> dict[str, int]:
        counts = {MemoryInfo.RAW: 0, MemoryInfo.OCCUPIED: 0, MemoryInfo.ENTANGLED: 0}
        for memory in memories:
            state = self._state(self.r2, memory)
            if state in counts:
                counts[state] += 1
        return counts

    def _record_state_trace(self) -> None:
        r2_memories = self.r2.get_components_by_type("MemoryArray")[0]
        left_counts = self._count_states(r2_memories[: self.left_capacity])
        right_counts = self._count_states(r2_memories[self.left_capacity : self.left_capacity + self.right_capacity])
        row = {
            "time_s": self.timeline.now() / PS_PER_S,
            "left_raw": left_counts[MemoryInfo.RAW],
            "left_occupied": left_counts[MemoryInfo.OCCUPIED],
            "left_entangled": left_counts[MemoryInfo.ENTANGLED],
            "right_raw": right_counts[MemoryInfo.RAW],
            "right_occupied": right_counts[MemoryInfo.OCCUPIED],
            "right_entangled": right_counts[MemoryInfo.ENTANGLED],
        }
        if not self.state_trace or any(self.state_trace[-1].get(k) != v for k, v in row.items() if k != "time_s"):
            self.state_trace.append(row)

    def _left_pairs(self):
        r1_memories = self.r1.get_components_by_type("MemoryArray")[0]
        r2_memories = self.r2.get_components_by_type("MemoryArray")[0]
        for idx in range(self.left_capacity):
            yield idx, self.r1, idx, r1_memories[idx], self.r2, idx, r2_memories[idx], self.m12, self.left_cfg

    def _right_pairs(self):
        r2_memories = self.r2.get_components_by_type("MemoryArray")[0]
        r3_memories = self.r3.get_components_by_type("MemoryArray")[0]
        for idx in range(self.right_capacity):
            yield idx, self.r2, self.left_capacity + idx, r2_memories[self.left_capacity + idx], self.r3, idx, r3_memories[idx], self.m23, self.right_cfg

    def _generation_active(self, pairs) -> bool:
        for _, ra, _, ma, rb, _, mb, _, _ in pairs:
            if _memory_has_generation_protocol(ma) or _memory_has_generation_protocol(mb):
                return True
        return False

    def _start_generation_attempts(self) -> None:
        self._start_link_generation(list(self._left_pairs()))
        self._start_link_generation(list(self._right_pairs()))

    def _record_elementary_arrivals(self) -> None:
        for memory in self.r2.get_components_by_type("MemoryArray")[0][: self.left_capacity]:
            info = self._info(self.r2, memory)
            key = (memory.name, info.entangle_time)
            if info.state == MemoryInfo.ENTANGLED and info.remote_node == self.r1.name and info.entangle_time >= 0 and key not in self._seen_elementary_arrivals:
                self._seen_elementary_arrivals.add(key)
                self.left_arrival_times_s.append(info.entangle_time / PS_PER_S)
        for memory in self.r2.get_components_by_type("MemoryArray")[0][self.left_capacity : self.left_capacity + self.right_capacity]:
            info = self._info(self.r2, memory)
            key = (memory.name, info.entangle_time)
            if info.state == MemoryInfo.ENTANGLED and info.remote_node == self.r3.name and info.entangle_time >= 0 and key not in self._seen_elementary_arrivals:
                self._seen_elementary_arrivals.add(key)
                self.right_arrival_times_s.append(info.entangle_time / PS_PER_S)

    def _start_link_generation(self, pairs) -> None:
        if not self.multiplexing and self._generation_active(pairs):
            return

        for idx, router_a, memory_index_a, memory_a, router_b, memory_index_b, memory_b, bsm, cfg in pairs:
            if self._state(router_a, memory_a) != MemoryInfo.RAW or self._state(router_b, memory_b) != MemoryInfo.RAW:
                continue
            self._start_generation_pair(router_a, memory_index_a, memory_a, router_b, memory_index_b, memory_b, bsm, cfg, idx)
            if not self.multiplexing:
                return

    def _start_generation_pair(self, router_a, memory_index_a: int, memory_a, router_b, memory_index_b: int, memory_b, bsm, cfg, index: int) -> None:
        self.protocol_counter += 1
        name_prefix = f"alloc.eg{self.protocol_counter}.{cfg.label}.{index}"
        router_a.resource_manager.memory_manager.update(memory_a, MemoryInfo.OCCUPIED)
        router_b.resource_manager.memory_manager.update(memory_b, MemoryInfo.OCCUPIED)
        proto_a, proto_b, _, _ = _install_generation_pair(
            router_a,
            router_b,
            bsm,
            cfg,
            memory_index_a,
            memory_index_b,
            name_prefix,
        )
        if proto_a.primary:
            proto_b.start()
            proto_a.start()
        else:
            proto_a.start()
            proto_b.start()

    def _middle_left_ready(self):
        memories = self.r2.get_components_by_type("MemoryArray")[0]
        ready = []
        for idx in range(self.left_capacity):
            memory = memories[idx]
            info = self._info(self.r2, memory)
            remote_memory = self.timeline.get_entity_by_name(info.remote_memo) if info.remote_memo else None
            remote_ready = remote_memory is not None and self._state(self.r1, remote_memory) == MemoryInfo.ENTANGLED
            if info.state == MemoryInfo.ENTANGLED and info.remote_node == self.r1.name and remote_ready:
                ready.append((info.entangle_time, idx, memory.name, memory))
        return [memory for _, _, _, memory in sorted(ready)]

    def _middle_right_ready(self):
        memories = self.r2.get_components_by_type("MemoryArray")[0]
        ready = []
        for idx in range(self.right_capacity):
            memory = memories[self.left_capacity + idx]
            info = self._info(self.r2, memory)
            remote_memory = self.timeline.get_entity_by_name(info.remote_memo) if info.remote_memo else None
            remote_ready = remote_memory is not None and self._state(self.r3, remote_memory) == MemoryInfo.ENTANGLED
            if info.state == MemoryInfo.ENTANGLED and info.remote_node == self.r3.name and remote_ready:
                ready.append((info.entangle_time, idx, memory.name, memory))
        return [memory for _, _, _, memory in sorted(ready)]

    def _start_ready_swaps(self) -> None:
        while True:
            left_ready = self._middle_left_ready()
            right_ready = self._middle_right_ready()
            if not left_ready or not right_ready:
                return
            left_memory, right_memory = self._select_swap_memories(left_ready, right_ready)
            if not self._schedule_swap(left_memory, right_memory):
                return

    def _select_swap_memories(self, left_ready, right_ready):
        if self.swap_selection_policy == "fifo":
            return left_ready[0], right_ready[0]
        generator = self.r2.get_generator()
        return generator.choice(left_ready), generator.choice(right_ready)

    def _schedule_swap(self, left_middle_memory, right_middle_memory) -> bool:
        left_end_memory = self.timeline.get_entity_by_name(left_middle_memory.entangled_memory["memo_id"])
        right_end_memory = self.timeline.get_entity_by_name(right_middle_memory.entangled_memory["memo_id"])
        if left_end_memory is None or right_end_memory is None:
            return False

        left_info = self._info(self.r2, left_middle_memory)
        right_info = self._info(self.r2, right_middle_memory)
        now_ps = self.timeline.now()
        self.left_waits_s.append(max(0.0, (now_ps - left_info.entangle_time) / PS_PER_S))
        self.right_waits_s.append(max(0.0, (now_ps - right_info.entangle_time) / PS_PER_S))

        self.swap_counter += 1
        start_rule_verified_bell_swap(
            self.r2,
            self.r1,
            self.r3,
            left_middle_memory,
            right_middle_memory,
            left_end_memory,
            right_end_memory,
            name_prefix=f"alloc.{self.swap_counter}",
            swap=self.swap_cfg,
        )
        return True

    def _consume_delivered_pairs(self) -> None:
        r1_memories = self.r1.get_components_by_type("MemoryArray")[0]
        for memory in list(r1_memories)[: self.left_capacity]:
            info = self._info(self.r1, memory)
            if info.state != MemoryInfo.ENTANGLED or info.remote_node != self.r3.name:
                continue
            fidelity = float(memory.fidelity)
            self.delivery_times_s.append(self.timeline.now() / PS_PER_S)
            self.delivery_fidelities.append(float(fidelity))
            if fidelity >= self.target_fidelity:
                self.accepted_delivery_times_s.append(self.timeline.now() / PS_PER_S)
                self.accepted_delivery_fidelities.append(float(fidelity))
            self._release_pair(self.r1, memory)

    def _release_pair(self, router, memory) -> None:
        remote_memory_name = memory.entangled_memory.get("memo_id")
        remote_memory = self.timeline.get_entity_by_name(remote_memory_name) if remote_memory_name else None
        router.resource_manager.update(None, memory, MemoryInfo.RAW)
        if remote_memory is not None:
            remote_router = self.memory_owner.get(remote_memory.name)
            if remote_router is not None:
                remote_router.resource_manager.update(None, remote_memory, MemoryInfo.RAW)


def _run_timeline_until_stop(timeline: Timeline, *, wall_timeout_s: float | None = None, max_events: int | None = None) -> dict[str, Any]:
    """Run a timeline with wall/event guards for pathological event cascades."""

    started = monotonic()
    timeline.is_running = True
    timed_out = False
    event_limited = False
    try:
        while len(timeline.events) > 0:
            if wall_timeout_s is not None and monotonic() - started >= wall_timeout_s:
                timed_out = True
                break
            if max_events is not None and timeline.run_counter >= max_events:
                event_limited = True
                break

            event = timeline.events.pop()
            if event.time >= timeline.stop_time:
                timeline.schedule(event)
                break
            if event.is_invalid():
                continue

            assert timeline.time <= event.time, f"invalid event time for process scheduled on {event.process.owner}"
            timeline.time = event.time
            event.process.run()
            timeline.run_counter += 1
    finally:
        timeline.is_running = False

    return {
        "timed_out": timed_out,
        "event_limited": event_limited,
        "wall_time_s": monotonic() - started,
        "timeline_now_s": timeline.now() / PS_PER_S,
        "run_counter": timeline.run_counter,
        "schedule_counter": timeline.schedule_counter,
        "pending_events": len(timeline.events),
    }


def run_memory_allocation_sequence_trial(
    config: AllocationQueueConfig,
    *,
    left_capacity: int,
    right_capacity: int,
    left_ttl_s: float,
    right_ttl_s: float,
    target_fidelity: float,
    multiplexing: bool,
    horizon_s: float,
    seed: int,
    control_period_s: float = 5e-5,
    wall_timeout_s: float | None = None,
    max_timeline_events: int | None = None,
    swap_selection_policy: str = "random",
) -> Dict[str, Any]:
    """Run one SeQUeNCe-backed continuous simulation for a memory allocation."""

    _use_pair_level_single_heralded()
    left_cfg = config.left_link_config()
    right_cfg = config.right_link_config()
    swap_cfg = config.swap_config()
    left_cfg = ElementaryLinkConfig(**{**left_cfg.__dict__, "generated_keepalive_s": left_ttl_s})
    right_cfg = ElementaryLinkConfig(**{**right_cfg.__dict__, "generated_keepalive_s": right_ttl_s})

    start_time_ps = max(
        int(1e9),
        5 * max(_classical_channel_delay_ps(left_cfg.distance_m), _classical_channel_delay_ps(right_cfg.distance_m)),
    )
    stop_time_ps = start_time_ps + int(horizon_s * PS_PER_S)
    timeline = Timeline(stop_time_ps, formalism=BELL_DIAGONAL_STATE_FORMALISM)
    timeline.swap_diagnostics = []
    _configure_generated_lifetime_controls(
        timeline,
        link_calibrations_by_middle={"m12": left_cfg, "m23": right_cfg},
        swap_calibration=swap_cfg,
        policy=EntanglementLifetimePolicy(),
    )
    r1 = ExtendedQuantumRouter("r1", timeline, memo_size=left_capacity, component_templates=left_cfg.component_templates(left_capacity), seed=seed)
    r2 = ExtendedQuantumRouter(
        "r2",
        timeline,
        memo_size=left_capacity + right_capacity,
        component_templates=left_cfg.component_templates(left_capacity + right_capacity),
        seed=seed + 1,
    )
    r3 = ExtendedQuantumRouter("r3", timeline, memo_size=right_capacity, component_templates=right_cfg.component_templates(right_capacity), seed=seed + 2)
    m12 = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=seed + 3, component_templates=left_cfg.bsm_templates())
    m23 = ExtendedBSMNode("m23", timeline, ["r2", "r3"], seed=seed + 4, component_templates=right_cfg.bsm_templates())

    r1.add_bsm_node(m12.name, r2.name)
    r2.add_bsm_node(m12.name, r1.name)
    r2.add_bsm_node(m23.name, r3.name)
    r3.add_bsm_node(m23.name, r2.name)

    classical_distances = _shortest_path_distances(
        [r1, r2, r3, m12, m23],
        [
            # Classical control links are parallel to the elementary quantum links:
            # router-to-router distance is the full link length, while each BSM leg is half.
            (r1.name, r2.name, left_cfg.distance_m),
            (r2.name, r3.name, right_cfg.distance_m),
            (r1.name, m12.name, left_cfg.bsm_segment_distance_m),
            (r2.name, m12.name, left_cfg.bsm_segment_distance_m),
            (r2.name, m23.name, right_cfg.bsm_segment_distance_m),
            (r3.name, m23.name, right_cfg.bsm_segment_distance_m),
        ],
    )
    _connect_classical_fully([r1, r2, r3, m12, m23], classical_distances, None)
    _make_quantum_channel("qc.r1.m12", timeline, r1, m12.name, left_cfg)
    _make_quantum_channel("qc.r2.m12", timeline, r2, m12.name, left_cfg)
    _make_quantum_channel("qc.r2.m23", timeline, r2, m23.name, right_cfg)
    _make_quantum_channel("qc.r3.m23", timeline, r3, m23.name, right_cfg)

    timeline.init()
    controller = _MemoryAllocationController(
        timeline=timeline,
        r1=r1,
        r2=r2,
        r3=r3,
        m12=m12,
        m23=m23,
        left_cfg=left_cfg,
        right_cfg=right_cfg,
        swap_cfg=swap_cfg,
        left_capacity=left_capacity,
        right_capacity=right_capacity,
        multiplexing=multiplexing,
        target_fidelity=target_fidelity,
        stop_time_ps=stop_time_ps,
        control_period_ps=max(1, int(control_period_s * PS_PER_S)),
        swap_selection_policy=swap_selection_policy,
    )
    timeline.schedule(Event(start_time_ps, Process(controller, "start", [])))
    run_status = _run_timeline_until_stop(
        timeline,
        wall_timeout_s=wall_timeout_s,
        max_events=max_timeline_events,
    )

    throughput_hz = len(controller.delivery_times_s) / horizon_s if horizon_s > 0 else 0.0
    return {
        "pair_count": len(controller.delivery_times_s),
        "throughput_hz": throughput_hz,
        "fidelity_mean": mean(controller.delivery_fidelities) if controller.delivery_fidelities else float("nan"),
        "accepted_pair_count": len(controller.accepted_delivery_times_s),
        "accepted_throughput_hz": len(controller.accepted_delivery_times_s) / horizon_s if horizon_s > 0 else 0.0,
        "accepted_fidelity_mean": mean(controller.accepted_delivery_fidelities) if controller.accepted_delivery_fidelities else float("nan"),
        "delivery_times_s": controller.delivery_times_s,
        "delivery_fidelities": controller.delivery_fidelities,
        "accepted_delivery_times_s": controller.accepted_delivery_times_s,
        "accepted_delivery_fidelities": controller.accepted_delivery_fidelities,
        "left_waits_s": controller.left_waits_s,
        "right_waits_s": controller.right_waits_s,
        "left_arrival_times_s": controller.left_arrival_times_s,
        "right_arrival_times_s": controller.right_arrival_times_s,
        "left_arrival_rate_hz": len(controller.left_arrival_times_s) / horizon_s if horizon_s > 0 else 0.0,
        "right_arrival_rate_hz": len(controller.right_arrival_times_s) / horizon_s if horizon_s > 0 else 0.0,
        "state_trace": controller.state_trace,
        "swap_diagnostics": list(timeline.swap_diagnostics),
        **run_status,
    }


def run_memory_allocation_queue_experiment(
    config: AllocationQueueConfig,
    *,
    left_capacity: int,
    right_capacity: int,
    required_fidelity: float,
    multiplexing: bool,
    num_runs: int = 20,
    horizon_s: float = 10.0,
    base_seed: int = 91_000,
    control_period_s: float = 1e-5,
    swap_selection_policy: str = "random",
    include_trials: bool = True,
) -> Dict[str, Any]:
    """Run independent queue simulations and summarize rate/fidelity."""

    terms = reference_model_terms(config)
    raw_pair_fidelity = terms["F_max"]
    left_ttl_s, right_ttl_s = reference_ttls_for_required_fidelity(config, required_fidelity)
    trials = [
        run_memory_allocation_sequence_trial(
            config,
            left_capacity=left_capacity,
            right_capacity=right_capacity,
            left_ttl_s=left_ttl_s,
            right_ttl_s=right_ttl_s,
            target_fidelity=required_fidelity,
            multiplexing=multiplexing,
            horizon_s=horizon_s,
            seed=base_seed + idx,
            control_period_s=control_period_s,
            swap_selection_policy=swap_selection_policy,
        )
        for idx in range(num_runs)
    ]
    rates = [trial["throughput_hz"] for trial in trials]
    accepted_rates = [trial["accepted_throughput_hz"] for trial in trials]
    fidelities = [trial["fidelity_mean"] for trial in trials if not isnan(trial["fidelity_mean"])]
    accepted_fidelities = [trial["accepted_fidelity_mean"] for trial in trials if not isnan(trial["accepted_fidelity_mean"])]
    delivered_fidelities = [fidelity for trial in trials for fidelity in trial["delivery_fidelities"]]
    raw_model_rate = allocation_model_rate(
        left_capacity,
        right_capacity,
        multiplexing=multiplexing,
        left_rate_hz=config.left_rate_hz,
        right_rate_hz=config.right_rate_hz,
        left_ttl_s=left_ttl_s,
        right_ttl_s=right_ttl_s,
        swap_time_s=terms["Tswp"],
        left_reset_s=terms["T1"],
        right_reset_s=terms["T2"],
    )
    model_rate = config.swap_success_prob * raw_model_rate
    model_fidelity = allocation_model_fidelity(
        config,
        left_capacity,
        right_capacity,
        multiplexing=multiplexing,
        left_ttl_s=left_ttl_s,
        right_ttl_s=right_ttl_s,
    )
    row = {
        "left_capacity": left_capacity,
        "right_capacity": right_capacity,
        "required_fidelity": required_fidelity,
        "multiplexing": multiplexing,
        "left_ttl_s": left_ttl_s,
        "right_ttl_s": right_ttl_s,
        "model_rate_hz": model_rate,
        "model_fidelity": model_fidelity,
        "raw_pair_fidelity": raw_pair_fidelity,
        "simulation_rate_hz": mean(rates) if rates else 0.0,
        "simulation_rate_std_hz": stdev(rates) if len(rates) > 1 else 0.0,
        "simulation_accepted_rate_hz": mean(accepted_rates) if accepted_rates else 0.0,
        "simulation_accepted_rate_std_hz": stdev(accepted_rates) if len(accepted_rates) > 1 else 0.0,
        "simulation_fidelity": mean(fidelities) if fidelities else float("nan"),
        "simulation_fidelity_std": stdev(fidelities) if len(fidelities) > 1 else 0.0,
        "simulation_fidelity_min": min(delivered_fidelities) if delivered_fidelities else float("nan"),
        "simulation_fidelity_max": max(delivered_fidelities) if delivered_fidelities else float("nan"),
        "simulation_accepted_fidelity": mean(accepted_fidelities) if accepted_fidelities else float("nan"),
        "simulation_below_target_count": sum(1 for fidelity in delivered_fidelities if fidelity < required_fidelity),
        "simulation_delivered_count": len(delivered_fidelities),
    }
    if include_trials:
        row["trials"] = trials
    return row


def _run_allocation_sweep_task(args: tuple[Any, ...]) -> Dict[str, Any]:
    (
        config,
        left_capacity,
        right_capacity,
        required_fidelity,
        multiplexing,
        num_runs,
        horizon_s,
        base_seed,
        control_period_s,
        swap_selection_policy,
        include_trials,
    ) = args
    return run_memory_allocation_queue_experiment(
        config,
        left_capacity=left_capacity,
        right_capacity=right_capacity,
        required_fidelity=required_fidelity,
        multiplexing=multiplexing,
        num_runs=num_runs,
        horizon_s=horizon_s,
        base_seed=base_seed,
        control_period_s=control_period_s,
        swap_selection_policy=swap_selection_policy,
        include_trials=include_trials,
    )


def _run_allocation_sweep_model_task(args: tuple[Any, ...]) -> Dict[str, Any]:
    config, left_capacity, right_capacity, required_fidelity, multiplexing = args
    return run_memory_allocation_queue_experiment(
        config,
        left_capacity=left_capacity,
        right_capacity=right_capacity,
        required_fidelity=required_fidelity,
        multiplexing=multiplexing,
        num_runs=0,
        horizon_s=0.0,
        base_seed=0,
    )


def _run_allocation_trial_task(args: tuple[Any, ...]) -> tuple[int, int, dict[str, Any]]:
    (
        point_idx,
        run_idx,
        config,
        left_capacity,
        right_capacity,
        required_fidelity,
        multiplexing,
        horizon_s,
        seed,
        control_period_s,
        swap_selection_policy,
    ) = args
    left_ttl_s, right_ttl_s = reference_ttls_for_required_fidelity(config, required_fidelity)
    trial = run_memory_allocation_sequence_trial(
        config,
        left_capacity=left_capacity,
        right_capacity=right_capacity,
        left_ttl_s=left_ttl_s,
        right_ttl_s=right_ttl_s,
        target_fidelity=required_fidelity,
        multiplexing=multiplexing,
        horizon_s=horizon_s,
        seed=seed,
        control_period_s=control_period_s,
        swap_selection_policy=swap_selection_policy,
    )
    return point_idx, run_idx, trial


def _summarize_allocation_point(
    model_row: dict[str, Any],
    trials: list[dict[str, Any]],
    *,
    include_trials: bool = True,
) -> dict[str, Any]:
    rates = [trial["throughput_hz"] for trial in trials]
    accepted_rates = [trial["accepted_throughput_hz"] for trial in trials]
    fidelities = [trial["fidelity_mean"] for trial in trials if not isnan(trial["fidelity_mean"])]
    accepted_fidelities = [trial["accepted_fidelity_mean"] for trial in trials if not isnan(trial["accepted_fidelity_mean"])]
    delivered_fidelities = [fidelity for trial in trials for fidelity in trial["delivery_fidelities"]]
    required_fidelity = float(model_row["required_fidelity"])
    timed_out_count = sum(1 for trial in trials if trial.get("timed_out") or trial.get("event_limited"))
    row = dict(model_row)
    row.update(
        {
            "simulation_rate_hz": mean(rates) if rates else 0.0,
            "simulation_rate_std_hz": stdev(rates) if len(rates) > 1 else 0.0,
            "simulation_accepted_rate_hz": mean(accepted_rates) if accepted_rates else 0.0,
            "simulation_accepted_rate_std_hz": stdev(accepted_rates) if len(accepted_rates) > 1 else 0.0,
            "simulation_fidelity": mean(fidelities) if fidelities else float("nan"),
            "simulation_fidelity_std": stdev(fidelities) if len(fidelities) > 1 else 0.0,
            "simulation_fidelity_min": min(delivered_fidelities) if delivered_fidelities else float("nan"),
            "simulation_fidelity_max": max(delivered_fidelities) if delivered_fidelities else float("nan"),
            "simulation_accepted_fidelity": mean(accepted_fidelities) if accepted_fidelities else float("nan"),
            "simulation_below_target_count": sum(1 for fidelity in delivered_fidelities if fidelity < required_fidelity),
            "simulation_delivered_count": len(delivered_fidelities),
            "timed_out_runs": timed_out_count,
        }
    )
    if include_trials:
        row["trials"] = trials
    return row


def run_allocation_sweep(
    config: AllocationQueueConfig,
    *,
    allocations: Sequence[Tuple[int, int]],
    required_fidelities: Iterable[float],
    multiplexing: bool,
    num_runs: int = 20,
    horizon_s: float = 10.0,
    base_seed: int = 91_000,
    max_workers: int | None = None,
    parallel_backend: str = "process",
    progress: bool = False,
    heartbeat_s: float = 60.0,
    parallelize_runs: bool = True,
    control_period_s: float = 1e-5,
    swap_selection_policy: str = "random",
    include_trials: bool = True,
) -> List[Dict[str, Any]]:
    """Evaluate all memory allocations and required fidelity points."""

    tasks = []
    task_labels = []
    for alloc_idx, (left_capacity, right_capacity) in enumerate(allocations):
        for fid_idx, required_fidelity in enumerate(required_fidelities):
            tasks.append(
                (
                    config,
                    left_capacity,
                    right_capacity,
                    float(required_fidelity),
                    multiplexing,
                    num_runs,
                    horizon_s,
                    base_seed + 10_000 * alloc_idx + 100 * fid_idx,
                    control_period_s,
                    swap_selection_policy,
                    include_trials,
                )
            )
            task_labels.append(f"allocation {left_capacity}:{right_capacity}, F_req={float(required_fidelity):.6f}")
    if max_workers is None:
        max_workers = min(len(tasks), max(1, (os.cpu_count() or 1) - 1))
    if num_runs <= 0:
        if progress:
            print(f"Running model-only sweep: {len(tasks)} points")
        rows = [
            _run_allocation_sweep_model_task((config, left_capacity, right_capacity, required_fidelity, multiplexing))
            for _, left_capacity, right_capacity, required_fidelity, multiplexing, *_ in tasks
        ]
        if progress:
            print("Completed model-only sweep")
        return rows
    if parallelize_runs:
        model_rows = [
            _run_allocation_sweep_model_task((config, left_capacity, right_capacity, required_fidelity, multiplexing))
            for _, left_capacity, right_capacity, required_fidelity, multiplexing, *_ in tasks
        ]
        trial_tasks = []
        trial_labels = []
        for point_idx, task in enumerate(tasks):
            _, left_capacity, right_capacity, required_fidelity, multiplexing, _, horizon_s, base_seed_for_point, control_period_s, swap_selection_policy = task
            for run_idx in range(num_runs):
                trial_tasks.append(
                    (
                        point_idx,
                        run_idx,
                        config,
                        left_capacity,
                        right_capacity,
                        required_fidelity,
                        multiplexing,
                        horizon_s,
                        base_seed_for_point + run_idx,
                        control_period_s,
                        swap_selection_policy,
                    )
                )
                trial_labels.append(f"{task_labels[point_idx]}, run {run_idx + 1}/{num_runs}")

        trials_by_point: dict[int, list[dict[str, Any]]] = {idx: [None] * num_runs for idx in range(len(tasks))}
        if max_workers <= 1 or len(trial_tasks) <= 1:
            total = len(trial_tasks)
            for idx, (trial_task, label) in enumerate(zip(trial_tasks, trial_labels), start=1):
                if progress:
                    print(f"[{idx}/{total}] starting {label}")
                point_idx, run_idx, trial = _run_allocation_trial_task(trial_task)
                trials_by_point[point_idx][run_idx] = trial
                if progress:
                    print(f"[{idx}/{total}] completed {label}")
        else:
            executor_cls = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
            if progress:
                print(f"Running sweep: {len(tasks)} points, {len(trial_tasks)} runs, workers={max_workers}, backend={parallel_backend}")
            with executor_cls(max_workers=max_workers) as executor:
                future_to_index = {executor.submit(_run_allocation_trial_task, task): idx for idx, task in enumerate(trial_tasks)}
                pending = set(future_to_index)
                completed = 0
                last_heartbeat = monotonic()
                while pending:
                    done, pending = wait(pending, timeout=heartbeat_s if progress else None, return_when=FIRST_COMPLETED)
                    if not done:
                        if progress:
                            running = [trial_labels[future_to_index[future]] for future in pending]
                            print(f"[{completed}/{len(trial_tasks)}] still running after {monotonic() - last_heartbeat:.0f}s: {running[:20]}{' ...' if len(running) > 20 else ''}")
                            last_heartbeat = monotonic()
                        continue
                    for future in done:
                        trial_idx = future_to_index[future]
                        try:
                            point_idx, run_idx, trial = future.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as exc:
                            raise RuntimeError(f"Allocation sweep failed for {trial_labels[trial_idx]}") from exc
                        trials_by_point[point_idx][run_idx] = trial
                        completed += 1
                        if progress:
                            print(f"[{completed}/{len(trial_tasks)}] completed {trial_labels[trial_idx]}")
        return [
            _summarize_allocation_point(model_rows[idx], trials_by_point[idx], include_trials=include_trials)
            for idx in range(len(tasks))
        ]
    if max_workers <= 1 or len(tasks) <= 1:
        rows = []
        total = len(tasks)
        for idx, (task, label) in enumerate(zip(tasks, task_labels), start=1):
            if progress:
                print(f"[{idx}/{total}] starting {label}")
            rows.append(_run_allocation_sweep_task(task))
            if progress:
                print(f"[{idx}/{total}] completed {label}")
        return rows

    executor_cls = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
    if progress:
        print(f"Running sweep: {len(tasks)} points, workers={max_workers}, backend={parallel_backend}")
    with executor_cls(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(_run_allocation_sweep_task, task): idx for idx, task in enumerate(tasks)}
        rows_by_index: dict[int, Dict[str, Any]] = {}
        completed = 0
        pending = set(future_to_index)
        last_heartbeat = monotonic()
        while pending:
            done, pending = wait(pending, timeout=heartbeat_s if progress else None, return_when=FIRST_COMPLETED)
            if not done:
                if progress:
                    running = [task_labels[future_to_index[future]] for future in pending]
                    print(f"[{completed}/{len(tasks)}] still running after {monotonic() - last_heartbeat:.0f}s: {running}")
                    last_heartbeat = monotonic()
                continue
            for future in done:
                idx = future_to_index[future]
                try:
                    rows_by_index[idx] = future.result()
                except BrokenProcessPool:
                    raise
                except Exception as exc:
                    label = task_labels[idx]
                    raise RuntimeError(f"Allocation sweep failed for {label}") from exc
                completed += 1
                if progress:
                    print(f"[{completed}/{len(tasks)}] completed {task_labels[idx]}")
        return [rows_by_index[idx] for idx in range(len(tasks))]


class _ElementaryRateController:
    def __init__(
        self,
        *,
        timeline: Timeline,
        router_a: ExtendedQuantumRouter,
        router_b: ExtendedQuantumRouter,
        bsm: ExtendedBSMNode,
        link_cfg: ElementaryLinkConfig,
        stop_time_ps: int,
        control_period_ps: int,
    ) -> None:
        self.timeline = timeline
        self.router_a = router_a
        self.router_b = router_b
        self.bsm = bsm
        self.link_cfg = link_cfg
        self.stop_time_ps = stop_time_ps
        self.control_period_ps = control_period_ps
        self.protocol_counter = 0
        self.arrival_times_s: list[float] = []
        self._seen: set[tuple[str, int]] = set()
        self._in_reaction = False
        self._reaction_scheduled = False
        self._install_resource_update_hooks()

    def start(self) -> None:
        self._request_reaction()

    def _schedule_tick(self, time_ps: int) -> None:
        if time_ps <= self.stop_time_ps:
            self.timeline.schedule(Event(time_ps, Process(self, "tick", [])))

    def _info(self, router, memory):
        return router.resource_manager.memory_manager.get_info_by_memory(memory)

    def tick(self) -> None:
        memory_a = self.router_a.get_components_by_type("MemoryArray")[0][0]
        memory_b = self.router_b.get_components_by_type("MemoryArray")[0][0]
        info_a = self._info(self.router_a, memory_a)
        if info_a.state == MemoryInfo.ENTANGLED:
            key = (memory_a.name, info_a.entangle_time)
            if key not in self._seen:
                self._seen.add(key)
                self.arrival_times_s.append(info_a.entangle_time / PS_PER_S)
            self.router_a.resource_manager.update(None, memory_a, MemoryInfo.RAW)
            self.router_b.resource_manager.update(None, memory_b, MemoryInfo.RAW)
            info_a = self._info(self.router_a, memory_a)

        if info_a.state == MemoryInfo.RAW and self._info(self.router_b, memory_b).state == MemoryInfo.RAW:
            self.protocol_counter += 1
            self.router_a.resource_manager.memory_manager.update(memory_a, MemoryInfo.OCCUPIED)
            self.router_b.resource_manager.memory_manager.update(memory_b, MemoryInfo.OCCUPIED)
            proto_a, proto_b, _, _ = _install_generation_pair(
                self.router_a,
                self.router_b,
                self.bsm,
                self.link_cfg,
                0,
                0,
                f"elementary_rate.eg{self.protocol_counter}",
            )
            if proto_a.primary:
                proto_b.start()
                proto_a.start()
            else:
                proto_a.start()
                proto_b.start()

    def _install_resource_update_hooks(self) -> None:
        for router in (self.router_a, self.router_b):
            original_update = router.resource_manager.update

            def wrapped_update(protocol, memory, state, *, _original_update=original_update):
                result = _original_update(protocol, memory, state)
                self._request_reaction()
                return result

            router.resource_manager.update = wrapped_update

    def _request_reaction(self) -> None:
        if self._reaction_scheduled or self.timeline.now() >= self.stop_time_ps:
            return
        self._reaction_scheduled = True
        self.timeline.schedule(Event(self.timeline.now(), Process(self, "_react_to_state_change", [])))

    def _react_to_state_change(self) -> None:
        self._reaction_scheduled = False
        if self._in_reaction or self.timeline.now() >= self.stop_time_ps:
            return
        self._in_reaction = True
        try:
            self.tick()
        finally:
            self._in_reaction = False


class _AllocatedElementaryRateController:
    def __init__(
        self,
        *,
        timeline: Timeline,
        router_a: ExtendedQuantumRouter,
        router_b: ExtendedQuantumRouter,
        bsm: ExtendedBSMNode,
        link_cfg: ElementaryLinkConfig,
        capacity: int,
        multiplexing: bool,
        stop_time_ps: int,
    ) -> None:
        self.timeline = timeline
        self.router_a = router_a
        self.router_b = router_b
        self.bsm = bsm
        self.link_cfg = link_cfg
        self.capacity = capacity
        self.multiplexing = multiplexing
        self.stop_time_ps = stop_time_ps
        self.protocol_counter = 0
        self.arrival_times_s: list[float] = []
        self._seen: set[tuple[str, int]] = set()
        self._in_reaction = False
        self._reaction_scheduled = False
        self._install_resource_update_hooks()

    def start(self) -> None:
        self._request_reaction()

    def _info(self, router, memory):
        return router.resource_manager.memory_manager.get_info_by_memory(memory)

    def _install_resource_update_hooks(self) -> None:
        for router in (self.router_a, self.router_b):
            original_update = router.resource_manager.update

            def wrapped_update(protocol, memory, state, *, _original_update=original_update):
                result = _original_update(protocol, memory, state)
                self._request_reaction()
                return result

            router.resource_manager.update = wrapped_update

    def _request_reaction(self) -> None:
        if self._reaction_scheduled or self.timeline.now() >= self.stop_time_ps:
            return
        self._reaction_scheduled = True
        self.timeline.schedule(Event(self.timeline.now(), Process(self, "_react_to_state_change", [])))

    def _generation_active(self) -> bool:
        memories_a = self.router_a.get_components_by_type("MemoryArray")[0]
        memories_b = self.router_b.get_components_by_type("MemoryArray")[0]
        for memory_a, memory_b in zip(memories_a[: self.capacity], memories_b[: self.capacity]):
            if self._info(self.router_a, memory_a).state == MemoryInfo.OCCUPIED:
                return True
            if self._info(self.router_b, memory_b).state == MemoryInfo.OCCUPIED:
                return True
        return False

    def _react_to_state_change(self) -> None:
        self._reaction_scheduled = False
        if self._in_reaction or self.timeline.now() >= self.stop_time_ps:
            return
        self._in_reaction = True
        try:
            memories_a = self.router_a.get_components_by_type("MemoryArray")[0]
            memories_b = self.router_b.get_components_by_type("MemoryArray")[0]

            for memory_a, memory_b in zip(memories_a[: self.capacity], memories_b[: self.capacity]):
                info_a = self._info(self.router_a, memory_a)
                if info_a.state != MemoryInfo.ENTANGLED:
                    continue
                key = (memory_a.name, info_a.entangle_time)
                if key not in self._seen:
                    self._seen.add(key)
                    self.arrival_times_s.append(info_a.entangle_time / PS_PER_S)
                self.router_a.resource_manager.update(None, memory_a, MemoryInfo.RAW)
                self.router_b.resource_manager.update(None, memory_b, MemoryInfo.RAW)

            if not self.multiplexing and self._generation_active():
                return

            for index, (memory_a, memory_b) in enumerate(zip(memories_a[: self.capacity], memories_b[: self.capacity])):
                if self._info(self.router_a, memory_a).state != MemoryInfo.RAW:
                    continue
                if self._info(self.router_b, memory_b).state != MemoryInfo.RAW:
                    continue

                self.protocol_counter += 1
                self.router_a.resource_manager.memory_manager.update(memory_a, MemoryInfo.OCCUPIED)
                self.router_b.resource_manager.memory_manager.update(memory_b, MemoryInfo.OCCUPIED)
                proto_a, proto_b, _, _ = _install_generation_pair(
                    self.router_a,
                    self.router_b,
                    self.bsm,
                    self.link_cfg,
                    index,
                    index,
                    f"allocated_elementary_rate.eg{self.protocol_counter}.{index}",
                )
                if proto_a.primary:
                    proto_b.start()
                    proto_a.start()
                else:
                    proto_a.start()
                    proto_b.start()

                if not self.multiplexing:
                    return
        finally:
            self._in_reaction = False


def run_allocated_elementary_rate_trial(
    link: ElementaryLinkConfig,
    *,
    capacity: int,
    multiplexing: bool,
    seed: int = 0,
    horizon_s: float = 1.0,
) -> dict[str, Any]:
    """Measure an isolated elementary-link arrival rate for an allocation policy."""

    _use_pair_level_single_heralded()
    timeline = Timeline(int((horizon_s + 0.05) * PS_PER_S), formalism=BELL_DIAGONAL_STATE_FORMALISM)
    _configure_generated_lifetime_controls(timeline, link_calibrations_by_middle={"m12": link})
    r1 = ExtendedQuantumRouter("r1", timeline, memo_size=capacity, component_templates=link.component_templates(capacity), seed=seed)
    r2 = ExtendedQuantumRouter("r2", timeline, memo_size=capacity, component_templates=link.component_templates(capacity), seed=seed + 1)
    bsm = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=seed + 2, component_templates=link.bsm_templates())
    r1.add_bsm_node(bsm.name, r2.name)
    r2.add_bsm_node(bsm.name, r1.name)
    classical_distances = _shortest_path_distances(
        [r1, r2, bsm],
        [
            (r1.name, r2.name, link.distance_m),
            (r1.name, bsm.name, link.bsm_segment_distance_m),
            (r2.name, bsm.name, link.bsm_segment_distance_m),
        ],
    )
    _connect_classical_fully([r1, r2, bsm], classical_distances, None)
    _make_quantum_channel("qc.r1.m12", timeline, r1, bsm.name, link)
    _make_quantum_channel("qc.r2.m12", timeline, r2, bsm.name, link)
    timeline.init()
    start_time_ps = max(int(1e9), 5 * _classical_channel_delay_ps(link.distance_m))
    controller = _AllocatedElementaryRateController(
        timeline=timeline,
        router_a=r1,
        router_b=r2,
        bsm=bsm,
        link_cfg=link,
        capacity=capacity,
        multiplexing=multiplexing,
        stop_time_ps=start_time_ps + int(horizon_s * PS_PER_S),
    )
    timeline.schedule(Event(start_time_ps, Process(controller, "start", [])))
    run_status = _run_timeline_until_stop(timeline)
    return {
        "arrival_count": len(controller.arrival_times_s),
        "arrival_rate_hz": len(controller.arrival_times_s) / horizon_s if horizon_s > 0 else 0.0,
        "arrival_times_s": controller.arrival_times_s,
        **run_status,
    }


def run_elementary_rate_trial(
    link: ElementaryLinkConfig,
    *,
    seed: int = 0,
    horizon_s: float = 1.0,
    control_period_s: float = 5e-5,
) -> dict[str, Any]:
    """Estimate elementary-link arrival rate using the shared SeQUeNCe generation backend."""

    _use_pair_level_single_heralded()
    timeline = Timeline(int((horizon_s + 0.05) * PS_PER_S), formalism=BELL_DIAGONAL_STATE_FORMALISM)
    _configure_generated_lifetime_controls(timeline, link_calibrations_by_middle={"m12": link})
    r1 = ExtendedQuantumRouter("r1", timeline, memo_size=1, component_templates=link.component_templates(1), seed=seed)
    r2 = ExtendedQuantumRouter("r2", timeline, memo_size=1, component_templates=link.component_templates(1), seed=seed + 1)
    bsm = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=seed + 2, component_templates=link.bsm_templates())
    r1.add_bsm_node(bsm.name, r2.name)
    r2.add_bsm_node(bsm.name, r1.name)
    classical_distances = _shortest_path_distances(
        [r1, r2, bsm],
        [
            (r1.name, r2.name, link.distance_m),
            (r1.name, bsm.name, link.bsm_segment_distance_m),
            (r2.name, bsm.name, link.bsm_segment_distance_m),
        ],
    )
    _connect_classical_fully([r1, r2, bsm], classical_distances, None)
    _make_quantum_channel("qc.r1.m12", timeline, r1, bsm.name, link)
    _make_quantum_channel("qc.r2.m12", timeline, r2, bsm.name, link)
    timeline.init()
    start_time_ps = max(int(1e9), 5 * _classical_channel_delay_ps(link.distance_m))
    controller = _ElementaryRateController(
        timeline=timeline,
        router_a=r1,
        router_b=r2,
        bsm=bsm,
        link_cfg=link,
        stop_time_ps=start_time_ps + int(horizon_s * PS_PER_S),
        control_period_ps=max(1, int(control_period_s * PS_PER_S)),
    )
    timeline.schedule(Event(start_time_ps, Process(controller, "start", [])))
    timeline.run()
    return {
        "arrival_count": len(controller.arrival_times_s),
        "arrival_rate_hz": len(controller.arrival_times_s) / horizon_s if horizon_s > 0 else 0.0,
        "arrival_times_s": controller.arrival_times_s,
    }


def _elementary_rate_task(args: tuple[ElementaryLinkConfig, int, float, float]) -> float:
    link, seed, horizon_s, control_period_s = args
    return run_elementary_rate_trial(
        link,
        seed=seed,
        horizon_s=horizon_s,
        control_period_s=control_period_s,
    )["arrival_rate_hz"]


def estimate_elementary_rates(
    config: AllocationQueueConfig,
    *,
    num_runs: int = 4,
    horizon_s: float = 1.0,
    base_seed: int = 81_000,
    control_period_s: float = 5e-5,
    max_workers: int | None = None,
    parallel_backend: str = "process",
) -> tuple[float, float]:
    """Estimate model lambdas from continuous elementary-link SeQUeNCe generation."""

    left_tasks = [
        (config.left_link_config(), base_seed + idx, horizon_s, control_period_s)
        for idx in range(num_runs)
    ]
    right_tasks = [
        (config.right_link_config(), base_seed + 10_000 + idx, horizon_s, control_period_s)
        for idx in range(num_runs)
    ]

    if max_workers is None or max_workers <= 1 or num_runs <= 1:
        left_rates = [_elementary_rate_task(task) for task in left_tasks]
        right_rates = [_elementary_rate_task(task) for task in right_tasks]
    else:
        executor_cls = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        left_rates = []
        right_rates = []
        with executor_cls(max_workers=max_workers) as executor:
            left_futures = [executor.submit(_elementary_rate_task, task) for task in left_tasks]
            right_futures = [executor.submit(_elementary_rate_task, task) for task in right_tasks]
            for future in left_futures:
                left_rates.append(future.result())
            for future in right_futures:
                right_rates.append(future.result())
    return mean(left_rates), mean(right_rates)


def build_sequence_estimated_reference_config(
    *,
    lambda_num_runs: int = 10,
    lambda_horizon_s: float = 10.0,
    lambda_base_seed: int = 81_000,
    lambda_control_period_s: float = 5e-5,
    lambda_max_workers: int | None = None,
    lambda_parallel_backend: str = "process",
    left_distance_km: float = 32.0,
    right_distance_km: float = 18.0,
    coherence_time_s: float = 0.1,
    swap_success_prob: float = 0.5,
) -> AllocationQueueConfig:
    """Build the reference allocation config used by the validation notebook.

    The rates are estimated from the same SeQUeNCe elementary-generation backend
    used by the end-to-end allocation simulations.  This avoids using stale
    placeholder rates such as 300/720 Hz when validating the queueing model.
    """

    config = AllocationQueueConfig(
        left_rate_hz=0.0,
        right_rate_hz=0.0,
        left_distance_km=left_distance_km,
        right_distance_km=right_distance_km,
        left_raw_fidelity=0.9,
        right_raw_fidelity=0.9,
        swap_degradation=1.0,
        coherence_time_s=coherence_time_s,
        swap_success_prob=swap_success_prob,
    )
    terms = reference_model_terms(config)
    config = AllocationQueueConfig(
        left_rate_hz=0.0,
        right_rate_hz=0.0,
        left_distance_km=config.left_distance_km,
        right_distance_km=config.right_distance_km,
        left_raw_fidelity=terms["f_raw1"],
        right_raw_fidelity=terms["f_raw2"],
        swap_degradation=config.swap_degradation,
        coherence_time_s=config.coherence_time_s,
        swap_success_prob=config.swap_success_prob,
        classical_message_overhead_s=config.classical_message_overhead_s,
        optic_depol_left=config.optic_depol_left,
        optic_depol_right=config.optic_depol_right,
        optical_coherence_length_km=config.optical_coherence_length_km,
        memory_decoherence_rates_hz=config.memory_decoherence_rates_hz,
        memory_noise_type=config.memory_noise_type,
    )
    left_lambda, right_lambda = estimate_elementary_rates(
        config,
        num_runs=lambda_num_runs,
        horizon_s=lambda_horizon_s,
        base_seed=lambda_base_seed,
        control_period_s=lambda_control_period_s,
        max_workers=lambda_max_workers,
        parallel_backend=lambda_parallel_backend,
    )
    return AllocationQueueConfig(
        left_rate_hz=left_lambda,
        right_rate_hz=right_lambda,
        left_distance_km=config.left_distance_km,
        right_distance_km=config.right_distance_km,
        left_raw_fidelity=config.left_raw_fidelity,
        right_raw_fidelity=config.right_raw_fidelity,
        swap_degradation=config.swap_degradation,
        coherence_time_s=config.coherence_time_s,
        swap_success_prob=config.swap_success_prob,
        classical_message_overhead_s=config.classical_message_overhead_s,
        optic_depol_left=config.optic_depol_left,
        optic_depol_right=config.optic_depol_right,
        optical_coherence_length_km=config.optical_coherence_length_km,
        memory_decoherence_rates_hz=config.memory_decoherence_rates_hz,
        memory_noise_type=config.memory_noise_type,
    )
