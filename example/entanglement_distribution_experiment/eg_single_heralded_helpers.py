from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from heapq import heappop
from math import isnan
from statistics import mean, stdev
from typing import Any

import numpy as np

from sequence.components.fiber_quantum_channel import FiberSection, FiberSpec, fiberQuantumChannel
from sequence.components.optical_channel import ClassicalChannel
from sequence.app.request_app import RequestApp
from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM, MICROSECOND, SINGLE_HERALDED, SPEED_OF_LIGHT
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
from sequence.entanglement_management.generation.single_heralded import SingleHeraldedA, SingleHeraldedB
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.entanglement_management.swapping import EntanglementSwappingMessage, SwappingMsgType
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.node import BSMNode, QuantumRouter
from sequence.utils import log


PS_PER_S = 10**12
PAIR_LEVEL_SINGLE_HERALDED = "sh_pair_level"


@dataclass
class ElementaryLinkConfig:
    label: str
    # Full elementary-link distance between the two end routers. The helper
    # places the BSM at the midpoint and uses half this distance per fiber arm.
    distance_m: float = 20_000
    attenuation_db_per_m: float = 0.0002
    coherence_time_s: float = 0.01
    cutoff_ratio: float = 1.0
    local_coherence_time_s: float | None = None
    remote_coherence_time_s: float | None = None
    memory_decoherence_errors: list[float] | None = None
    optic_depol: float | None = None
    optic_depha: float | None = None
    memory_frequency_hz: float = 8e7
    memory_efficiency: float = 1.0
    bsm_success_rate: float = 0.5
    raw_fidelity: float = 0.9
    raw_epr_errors: list[float] = field(default_factory=lambda: [1 / 3, 1 / 3, 1 / 3])
    generated_keepalive_s: float | None = None
    detector_efficiency: float = 1.0
    detector_count_rate: float = 1e12
    detector_resolution_ps: int = 100
    classical_delay_ps: int | None = None
    fiber_spec: FiberSpec | None = None
    fiber_sections: list[FiberSection] | None = None
    swap_node_name: str | None = None

    @property
    def elementary_distance_m(self) -> float:
        return self.distance_m

    @property
    def bsm_segment_distance_m(self) -> float:
        return self.distance_m / 2

    def component_templates(self, memo_size: int) -> dict[str, Any]:
        return {
            "MemoryArray": {
                "fidelity": self.raw_fidelity,
                "frequency": self.memory_frequency_hz,
                "efficiency": self.memory_efficiency,
                "coherence_time": self.coherence_time_s,
                "decoherence_errors": self.memory_decoherence_errors,
                "cutoff_ratio": self.cutoff_ratio,
            }
        }

    def bsm_templates(self) -> dict[str, Any]:
        detector_template = {
            "efficiency": self.detector_efficiency,
            "count_rate": self.detector_count_rate,
            "time_resolution": self.detector_resolution_ps,
        }
        return {
            "encoding_type": "single_heralded",
            "SingleHeraldedBSM": {
                "success_rate": self.bsm_success_rate,
                "detectors": [detector_template.copy(), detector_template.copy()],
            },
        }



@dataclass
class SwapCalibration:
    success_prob: float = 1.0
    degradation: float = 0.95
    generated_keepalive_s: float | None = None


class EntanglementLifetimePolicy:
    """Placeholder hook for externally assigning a shorter post-generation lifetime."""

    def elementary_expire_time_ps(
        self,
        *,
        timeline: Timeline,
        local_memory,
        remote_memory,
        link_cfg: ElementaryLinkConfig | None,
        now_ps: int,
        default_expire_time_ps: int,
    ) -> int | None:
        return None

    def elementary_expire_times_ps(
        self,
        *,
        timeline: Timeline,
        local_memory,
        remote_memory,
        link_cfg: ElementaryLinkConfig | None,
        now_ps: int,
        default_local_expire_time_ps: int,
        default_remote_expire_time_ps: int,
    ) -> tuple[int, int] | None:
        return None

    def swapped_expire_time_ps(
        self,
        *,
        timeline: Timeline,
        local_memory,
        remote_memory,
        left_memory,
        right_memory,
        swap_cfg: SwapCalibration | None,
        now_ps: int,
        default_expire_time_ps: int,
    ) -> int | None:
        return None


class FixedGeneratedLifetimePolicy(EntanglementLifetimePolicy):
    """Simple placeholder policy that caps generated-pair lifetime after creation."""

    def elementary_expire_time_ps(
        self,
        *,
        timeline: Timeline,
        local_memory,
        remote_memory,
        link_cfg: ElementaryLinkConfig | None,
        now_ps: int,
        default_expire_time_ps: int,
    ) -> int | None:
        if link_cfg is None or link_cfg.generated_keepalive_s is None:
            return None
        return now_ps + int(round(link_cfg.generated_keepalive_s * PS_PER_S))

    def swapped_expire_time_ps(
        self,
        *,
        timeline: Timeline,
        local_memory,
        remote_memory,
        left_memory,
        right_memory,
        swap_cfg: SwapCalibration | None,
        now_ps: int,
        default_expire_time_ps: int,
    ) -> int | None:
        if swap_cfg is None or swap_cfg.generated_keepalive_s is None:
            return None
        return now_ps + int(round(swap_cfg.generated_keepalive_s * PS_PER_S))


class HeraldedFiberChannel(fiberQuantumChannel):
    """fiberQuantumChannel plus stock support for heralded memory photons."""

    def transmit(self, qubit, source) -> None:
        assert self.delay >= 0 and self.loss <= 1, f"QuantumChannel init() function has not been run for {self.name}"
        assert source == self.sender

        if len(self.send_bins) > 0:
            time = -1
            while time < self.timeline.now():
                time_bin = heappop(self.send_bins)
                time = self.timebin_to_time(time_bin, self.frequency)
            assert time == self.timeline.now(), f"qc {self.name} transmit method called at invalid time"

        if qubit.encoding_type["name"] == "fock":
            self.timeline.quantum_manager.add_loss(qubit.quantum_state, self.loss)
            future_time = self.timeline.now() + self.delay
            self.timeline.schedule(Event(future_time, Process(self.receiver, "receive_qubit", [source.name, qubit])))
            return

        if not ((self.sender.get_generator().random() > self.loss) or qubit.is_null):
            return

        if self._receiver_on_other_tl():
            self.timeline.quantum_manager.move_manage_to_server(qubit.quantum_state)

        if qubit.is_null:
            qubit.add_loss(self.loss)

        extra_delay = 0
        base_ps = self.delay
        if qubit.encoding_type["name"] == "polarization" and not qubit.is_null and self.J_total is not None:
            self._apply_jones(qubit, self.J_total)
            extra_delay += self._sample_pmd_delay_picoseconds(qubit)
            extra_delay += self._chromatic_delay_picoseconds(qubit)
            if self.base_group_delay_s:
                base_ps = int(round(self.base_group_delay_s * 1e12))
        future_time = max(self.timeline.now(), int(round(self.timeline.now() + base_ps + extra_delay)))
        self.timeline.schedule(Event(future_time, Process(self.receiver, "receive_qubit", [source.name, qubit])))


class ExtendedBSMNode(BSMNode):
    """BSM node extension for Raman-noise delivery from the custom fiber model."""

    def receive_noise_photon(self) -> None:
        bsm = self.components[self.first_component_name]
        detectors = getattr(bsm, "detectors", None)
        if not detectors:
            return
        detector = detectors[self.get_generator().integers(len(detectors))]
        bsm.trigger(detector, {"time": self.timeline.now()})


class ContinuousRequestApp(RequestApp):
    """Request app that consumes delivered end-to-end pairs immediately over one long reservation window."""

    def __init__(self, node: QuantumRouter, count_deliveries: bool):
        super().__init__(node)
        self.count_deliveries = count_deliveries
        self.delivery_times_s: list[float] = []
        self.delivery_fidelities: list[float] = []
        self.delivery_intervals_s: list[float] = []
        self.approved = False

    def get_reservation_result(self, reservation, result: bool) -> None:
        self.approved = result
        super().get_reservation_result(reservation, result)

    def get_memory(self, info) -> None:
        if info.state != "ENTANGLED":
            return
        if info.index not in self.memo_to_reservation:
            return

        reservation = self.memo_to_reservation[info.index]
        pair_fidelity = _memory_fidelity(info.memory)
        if pair_fidelity < reservation.fidelity:
            return

        now_s = self.node.timeline.now() / PS_PER_S
        if info.remote_node == reservation.responder:
            if self.count_deliveries:
                self.memory_counter += 1
                self.delivery_times_s.append(now_s)
                self.delivery_fidelities.append(pair_fidelity)
                previous_time_s = (reservation.start_time / PS_PER_S) if len(self.delivery_times_s) == 1 else self.delivery_times_s[-2]
                self.delivery_intervals_s.append(now_s - previous_time_s)
            self.node.resource_manager.update(None, info.memory, "RAW")
        elif info.remote_node == reservation.initiator:
            self.node.resource_manager.update(None, info.memory, "RAW")

    def cumulative_rate_trace(self, horizon_s: float, sample_period_s: float = 1.0) -> dict[str, list[float]]:
        """Return delivered-pair rate sampled at regular elapsed-time checkpoints."""

        if horizon_s <= 0 or sample_period_s <= 0:
            return {"times_s": [], "pair_counts": [], "rates_hz": []}
        checkpoints = np.arange(sample_period_s, horizon_s + sample_period_s * 0.5, sample_period_s)
        checkpoints = checkpoints[checkpoints <= horizon_s + 1e-12]
        deliveries = np.array(self.delivery_times_s, dtype=float)
        start_s = 0.0
        if checkpoints.size and self.delivery_intervals_s:
            start_s = max(0.0, self.delivery_times_s[0] - self.delivery_intervals_s[0])
        elapsed = checkpoints
        absolute_checkpoints = start_s + checkpoints
        counts = [int(np.count_nonzero(deliveries <= checkpoint)) for checkpoint in absolute_checkpoints]
        rates = [count / time_s if time_s > 0 else 0.0 for count, time_s in zip(counts, elapsed)]
        return {
            "times_s": elapsed.tolist(),
            "pair_counts": counts,
            "rates_hz": rates,
        }


class TraceRequestApp(RequestApp):
    """Request app that records the first successful delivery time without consuming the pair."""

    def __init__(self, node: QuantumRouter):
        super().__init__(node)
        self.first_delivery_time_ps: int | None = None
        self.first_delivery_memory = None
        self.first_delivery_fidelity: float | None = None
        self.deliveries: list[dict[str, Any]] = []

    def get_memory(self, info) -> None:
        if info.state != "ENTANGLED":
            return
        samplers = getattr(self.node.timeline, "runtime_pair_samplers", {})
        sampler = samplers.get(info.memory.name)
        if sampler is not None:
            sampler.sample("confirm")
        delivery = {
            "time_ps": self.node.timeline.now(),
            "memory": info.memory,
            "pair_fidelity": _memory_fidelity(info.memory),
            "reported_fidelity": float(info.memory.fidelity),
            "generation_time_ps": info.memory.generation_time,
            "expire_time_ps": _default_memory_expire_time_ps(info.memory, self.node.timeline.now()),
            "remote_node": info.remote_node,
            "remote_memo_id": info.remote_memo,
        }
        self.deliveries.append(delivery)
        if self.first_delivery_time_ps is None:
            self.first_delivery_time_ps = delivery["time_ps"]
            self.first_delivery_memory = info.memory
            self.first_delivery_fidelity = delivery["pair_fidelity"]


class RuntimePairSampler:
    """Samples memory snapshot and pair fidelity on scheduled timeline events."""

    def __init__(self, memory, horizon_s: float, sample_points: int):
        self.memory = memory
        self.timeline = memory.timeline
        self.horizon_ps = int(round(horizon_s * PS_PER_S))
        self.sample_points = max(2, sample_points)
        self.armed = False
        self.start_ps: int | None = None
        self.sample_times_ps: list[int] = []
        self.memory_fidelities: list[float] = []
        self.pair_fidelities: list[float] = []
        self.pair_states: list[list[float] | None] = []
        self.state_update_times_ps: list[int | None] = []
        self.event_samples: dict[str, list[dict[str, Any]]] = {}

    def arm_from_excitation(self, start_ps: int) -> None:
        if self.armed:
            return
        self.armed = True
        self.start_ps = start_ps
        end_ps = start_ps + self.horizon_ps
        schedule_times = np.linspace(start_ps, end_ps, self.sample_points)
        for sample_ps_float in schedule_times:
            sample_ps = int(round(sample_ps_float))
            self.timeline.schedule(Event(sample_ps, Process(self, "sample", [])))

    def schedule_absolute(self, start_ps: int, end_ps: int) -> None:
        if self.armed:
            return
        self.armed = True
        self.start_ps = start_ps
        schedule_times = np.linspace(start_ps, end_ps, self.sample_points)
        for sample_ps_float in schedule_times:
            sample_ps = int(round(sample_ps_float))
            self.timeline.schedule(Event(sample_ps, Process(self, "sample", [])))

    def sample(self, label: str | None = None) -> None:
        now_ps = self.timeline.now()
        memory_fidelity = float(self.memory.fidelity) if self.memory.entangled_memory["node_id"] is not None else 0.0
        pair_fidelity = _virtual_pair_fidelity(self.memory, now_ps)
        pair_state = list(self.timeline.quantum_manager.get(self.memory.qstate_key).state) if self.memory.qstate_key in self.timeline.quantum_manager.states else None
        state_update_time = int(self.memory.last_update_time) if pair_state is not None else None

        self.sample_times_ps.append(now_ps)
        self.memory_fidelities.append(memory_fidelity)
        self.pair_fidelities.append(pair_fidelity)
        self.pair_states.append(pair_state)
        self.state_update_times_ps.append(state_update_time)
        if label is not None:
            self.event_samples.setdefault(label, []).append(
                {
                    "time_ps": now_ps,
                    "memory_fidelity": memory_fidelity,
                    "pair_fidelity": pair_fidelity,
                    "pair_state": pair_state,
                    "state_update_time_ps": state_update_time,
                }
            )


def _default_memory_expire_time_ps(memory, now_ps: int) -> int:
    expire_time_ps = memory.get_expire_time()
    if expire_time_ps == float("inf"):
        return now_ps
    return int(expire_time_ps)


def _node_name_from_memory(memory) -> str | None:
    if memory is None or not getattr(memory, "name", None):
        return None
    return str(memory.name).split(".MemoryArray", 1)[0]


def _classical_delay_between_nodes_ps(timeline: Timeline, src_node_name: str | None, dst_node_name: str | None) -> int:
    if not src_node_name or not dst_node_name or src_node_name == dst_node_name:
        return 0
    src_node = timeline.get_entity_by_name(src_node_name)
    if src_node is None:
        return 0
    channel = getattr(src_node, "cchannels", {}).get(dst_node_name)
    if channel is None:
        return 0
    return int(channel.delay)


def _elementary_expire_times_with_endpoint_margin_ps(
    *,
    timeline: Timeline,
    local_memory,
    remote_memory,
    link_cfg: ElementaryLinkConfig | None,
    now_ps: int,
    default_local_expire_time_ps: int,
    default_remote_expire_time_ps: int,
) -> tuple[int, int] | None:
    if link_cfg is None or link_cfg.generated_keepalive_s is None:
        return None

    ttl_ps = int(round(link_cfg.generated_keepalive_s * PS_PER_S))
    strict_expire_ps = now_ps + ttl_ps
    local_node = _node_name_from_memory(local_memory)
    remote_node = _node_name_from_memory(remote_memory)
    swap_node = link_cfg.swap_node_name
    if swap_node is None and getattr(timeline, "swap_calibration", None) is not None:
        endpoint_nodes = {local_node, remote_node}
        if "r2" in endpoint_nodes:
            swap_node = "r2"

    if swap_node is None:
        return (
            int(max(now_ps, min(default_local_expire_time_ps, strict_expire_ps))),
            int(max(now_ps, min(default_remote_expire_time_ps, strict_expire_ps))),
        )

    local_margin_ps = 0 if local_node == swap_node else _classical_delay_between_nodes_ps(timeline, swap_node, local_node)
    remote_margin_ps = 0 if remote_node == swap_node else _classical_delay_between_nodes_ps(timeline, swap_node, remote_node)
    return (
        int(max(now_ps, min(default_local_expire_time_ps, strict_expire_ps + local_margin_ps))),
        int(max(now_ps, min(default_remote_expire_time_ps, strict_expire_ps + remote_margin_ps))),
    )


def _resolve_generated_expire_time_ps(
    *,
    timeline: Timeline,
    stage: str,
    now_ps: int,
    default_expire_time_ps: int,
    local_memory,
    remote_memory,
    link_cfg: ElementaryLinkConfig | None = None,
    swap_cfg: SwapCalibration | None = None,
    left_memory=None,
    right_memory=None,
) -> int:
    policy = getattr(timeline, "entanglement_lifetime_policy", None)
    candidate_expire_time_ps: int | None = None
    if policy is not None:
        if stage == "elementary" and hasattr(policy, "elementary_expire_time_ps"):
            candidate_expire_time_ps = policy.elementary_expire_time_ps(
                timeline=timeline,
                local_memory=local_memory,
                remote_memory=remote_memory,
                link_cfg=link_cfg,
                now_ps=now_ps,
                default_expire_time_ps=default_expire_time_ps,
            )
        elif stage == "swapped" and hasattr(policy, "swapped_expire_time_ps"):
            candidate_expire_time_ps = policy.swapped_expire_time_ps(
                timeline=timeline,
                local_memory=local_memory,
                remote_memory=remote_memory,
                left_memory=left_memory,
                right_memory=right_memory,
                swap_cfg=swap_cfg,
                now_ps=now_ps,
                default_expire_time_ps=default_expire_time_ps,
            )

    if candidate_expire_time_ps is None:
        if stage == "elementary" and link_cfg is not None and link_cfg.generated_keepalive_s is not None:
            candidate_expire_time_ps = now_ps + int(round(link_cfg.generated_keepalive_s * PS_PER_S))
        elif stage == "swapped" and swap_cfg is not None and swap_cfg.generated_keepalive_s is not None:
            candidate_expire_time_ps = now_ps + int(round(swap_cfg.generated_keepalive_s * PS_PER_S))

    if candidate_expire_time_ps is None:
        return int(default_expire_time_ps)

    return int(max(now_ps, min(int(default_expire_time_ps), int(candidate_expire_time_ps))))


@EntanglementGenerationA.register(PAIR_LEVEL_SINGLE_HERALDED)
class PairLevelSingleHeraldedA(SingleHeraldedA):
    """Additive single-heralded variant with pair-level observation support."""

    def _link_calibration(self) -> ElementaryLinkConfig | None:
        return getattr(self.owner.timeline, "link_calibrations_by_middle", {}).get(self.middle)

    def update_memory(self) -> bool | None:
        if self not in self.owner.protocols:
            return

        self.ent_round += 1
        if self.ent_round == 1:
            return True

        if self.ent_round == 2:
            if self.bsm_res[0] >= 1 and self.bsm_res[1] >= 1:
                quantum_manager = self.owner.timeline.quantum_manager
                remote_memory = self.owner.timeline.get_entity_by_name(self.remote_memo_id)
                keys = [self._qstate_key, remote_memory.qstate_key]

                if self._qstate_key not in quantum_manager.states:
                    in_fidelity = 1 - self.raw_fidelity
                    x_elem, y_elem, z_elem = (error * in_fidelity for error in self.raw_epr_errors)
                    state = [self.raw_fidelity, z_elem, x_elem, y_elem]
                    quantum_manager.set(keys, state)
                    now_ps = self.owner.timeline.now()
                    self.memory.last_update_time = now_ps
                    remote_memory.last_update_time = now_ps

                self._entanglement_succeed()
            else:
                self._entanglement_fail()
                return False

        return True

    def emit_event(self) -> None:
        super().emit_event()
        if self.ent_round == 1 and self.memory.generation_time >= 0:
            samplers = getattr(self.owner.timeline, "runtime_pair_samplers", {})
            sampler = samplers.get(self.memory.name)
            if sampler is not None:
                sampler.arm_from_excitation(self.memory.generation_time)

    def _entanglement_succeed(self):
        from sequence.resource_management.memory_manager import MemoryInfo

        log.logger.info(f"{self.owner.name} successful entanglement of memory {self.memory}")
        remote_memory = self.owner.timeline.get_entity_by_name(self.remote_memo_id)
        self.memory.entangled_memory["node_id"] = self.remote_node_name
        self.memory.entangled_memory["memo_id"] = self.remote_memo_id
        self.memory.fidelity = self.raw_fidelity

        now_ps = self.owner.timeline.now()
        default_expire_time_ps = _default_memory_expire_time_ps(self.memory, now_ps)
        remote_default_expire_time_ps = _default_memory_expire_time_ps(remote_memory, now_ps) if remote_memory is not None else default_expire_time_ps
        link_cfg = self._link_calibration()
        expire_times = None
        policy = getattr(self.owner.timeline, "entanglement_lifetime_policy", None)
        if policy is not None and hasattr(policy, "elementary_expire_times_ps"):
            expire_times = policy.elementary_expire_times_ps(
                timeline=self.owner.timeline,
                local_memory=self.memory,
                remote_memory=remote_memory,
                link_cfg=link_cfg,
                now_ps=now_ps,
                default_local_expire_time_ps=default_expire_time_ps,
                default_remote_expire_time_ps=remote_default_expire_time_ps,
            )
        if expire_times is None:
            expire_times = _elementary_expire_times_with_endpoint_margin_ps(
                timeline=self.owner.timeline,
                local_memory=self.memory,
                remote_memory=remote_memory,
                link_cfg=link_cfg,
                now_ps=now_ps,
                default_local_expire_time_ps=default_expire_time_ps,
                default_remote_expire_time_ps=remote_default_expire_time_ps,
            )
        if expire_times is None:
            expire_time_ps = _resolve_generated_expire_time_ps(
                timeline=self.owner.timeline,
                stage="elementary",
                now_ps=now_ps,
                default_expire_time_ps=default_expire_time_ps,
                local_memory=self.memory,
                remote_memory=remote_memory,
                link_cfg=link_cfg,
            )
            remote_expire_time_ps = expire_time_ps
        else:
            expire_time_ps, remote_expire_time_ps = expire_times

        self.memory.update_expire_time(expire_time_ps)
        if remote_memory is not None:
            remote_memory.update_expire_time(remote_expire_time_ps)

        self.update_resource_manager(self.memory, MemoryInfo.ENTANGLED)
        samplers = getattr(self.owner.timeline, "runtime_pair_samplers", {})
        sampler = samplers.get(self.memory.name)
        if sampler is not None:
            sampler.sample("herald")


@EntanglementGenerationB.register(PAIR_LEVEL_SINGLE_HERALDED)
class PairLevelSingleHeraldedB(SingleHeraldedB):
    """Factory registration shim for the additive pair-level single-heralded mode."""


class BellDiagonalSwappingA(EntanglementSwappingA):
    """Swapping protocol that stays additive and handles Bell-diagonal states."""

    @staticmethod
    def _refresh_memory_fidelity(memory) -> float:
        """Refresh a memory from its Bell-diagonal state before swap-time fidelity use."""

        try:
            memory.bds_decohere()
            memory.fidelity = float(memory.timeline.quantum_manager.get(memory.qstate_key).state[0])
        except Exception:
            memory.fidelity = float(memory.fidelity)
        return memory.fidelity

    def start(self) -> None:
        now_ps = self.owner.timeline.now()
        swap_cfg = getattr(self.owner.timeline, "swap_calibration", None)
        left_snapshot = float(self.left_memo.fidelity)
        right_snapshot = float(self.right_memo.fidelity)
        left_wait_s = max(0.0, (now_ps - self.left_memo.last_update_time) / PS_PER_S)
        right_wait_s = max(0.0, (now_ps - self.right_memo.last_update_time) / PS_PER_S)
        left_fidelity = self._refresh_memory_fidelity(self.left_memo)
        right_fidelity = self._refresh_memory_fidelity(self.right_memo)

        assert left_fidelity > 0 and right_fidelity > 0
        assert self.left_memo.entangled_memory["node_id"] == self.left_node
        assert self.right_memo.entangled_memory["node_id"] == self.right_node

        samplers = getattr(self.owner.timeline, "runtime_pair_samplers", {})
        for memo in (self.left_memo, self.right_memo):
            remote_memo = self.owner.timeline.get_entity_by_name(memo.entangled_memory["memo_id"])
            if remote_memo is not None:
                sampler = samplers.get(remote_memo.name)
                if sampler is not None:
                    sampler.sample("swap")

        left_end_memo = self.owner.timeline.get_entity_by_name(self.left_memo.entangled_memory["memo_id"])
        right_end_memo = self.owner.timeline.get_entity_by_name(self.right_memo.entangled_memory["memo_id"])

        if self.owner.get_generator().random() < self.success_probability():
            fidelity = self.updated_fidelity(left_fidelity, right_fidelity)
            self.is_success = True
            left_default_expire_time = (
                left_end_memo.get_expire_time() if left_end_memo is not None else self.left_memo.get_expire_time()
            )
            right_default_expire_time = (
                right_end_memo.get_expire_time() if right_end_memo is not None else self.right_memo.get_expire_time()
            )
            left_msg_expire_time = _resolve_generated_expire_time_ps(
                timeline=self.owner.timeline,
                stage="swapped",
                now_ps=now_ps,
                default_expire_time_ps=left_default_expire_time,
                local_memory=self.left_memo,
                remote_memory=self.right_memo,
                left_memory=self.left_memo,
                right_memory=self.right_memo,
                swap_cfg=swap_cfg,
            )
            right_msg_expire_time = _resolve_generated_expire_time_ps(
                timeline=self.owner.timeline,
                stage="swapped",
                now_ps=now_ps,
                default_expire_time_ps=right_default_expire_time,
                local_memory=self.left_memo,
                remote_memory=self.right_memo,
                left_memory=self.left_memo,
                right_memory=self.right_memo,
                swap_cfg=swap_cfg,
            )
            expire_time = min(left_msg_expire_time, right_msg_expire_time)

            msg_l = EntanglementSwappingMessage(
                SwappingMsgType.SWAP_RES,
                self.left_protocol_name,
                fidelity=fidelity,
                remote_node=self.right_memo.entangled_memory["node_id"],
                remote_memo=self.right_memo.entangled_memory["memo_id"],
                expire_time=left_msg_expire_time,
                meas_res=[],
            )
            msg_r = EntanglementSwappingMessage(
                SwappingMsgType.SWAP_RES,
                self.right_protocol_name,
                fidelity=fidelity,
                remote_node=self.left_memo.entangled_memory["node_id"],
                remote_memo=self.left_memo.entangled_memory["memo_id"],
                expire_time=right_msg_expire_time,
                meas_res=[],
            )
        else:
            msg_l = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES, self.left_protocol_name, fidelity=0)
            msg_r = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES, self.right_protocol_name, fidelity=0)

        self.owner.send_message(self.left_node, msg_l)
        self.owner.send_message(self.right_node, msg_r)
        diagnostics = getattr(self.owner.timeline, "swap_diagnostics", None)
        if diagnostics is not None:
            left_info = self.owner.resource_manager.memory_manager.get_info_by_memory(self.left_memo)
            right_info = self.owner.resource_manager.memory_manager.get_info_by_memory(self.right_memo)
            diagnostics.append(
                {
                    "time_ps": now_ps,
                    "left_start_ps": self.left_memo.generation_time,
                    "right_start_ps": self.right_memo.generation_time,
                    "left_memo_name": self.left_memo.name,
                    "right_memo_name": self.right_memo.name,
                    "left_herald_ps": left_info.entangle_time,
                    "right_herald_ps": right_info.entangle_time,
                    "left_wait_s": left_wait_s,
                    "right_wait_s": right_wait_s,
                    "max_wait_s": max(left_wait_s, right_wait_s),
                    "wait_imbalance_s": abs(left_wait_s - right_wait_s),
                    "left_remote_node": self.left_memo.entangled_memory["node_id"],
                    "right_remote_node": self.right_memo.entangled_memory["node_id"],
                    "left_expire_time_ps": int(self.left_memo.get_expire_time()),
                    "right_expire_time_ps": int(self.right_memo.get_expire_time()),
                    "left_endpoint_expire_time_ps": int(left_end_memo.get_expire_time()) if left_end_memo is not None else None,
                    "right_endpoint_expire_time_ps": int(right_end_memo.get_expire_time()) if right_end_memo is not None else None,
                    "left_msg_expire_time_ps": int(msg_l.expire_time) if self.is_success else now_ps,
                    "right_msg_expire_time_ps": int(msg_r.expire_time) if self.is_success else now_ps,
                    "left_snapshot_fidelity": left_snapshot,
                    "right_snapshot_fidelity": right_snapshot,
                    "left_refreshed_fidelity": left_fidelity,
                    "right_refreshed_fidelity": right_fidelity,
                    "swapped_fidelity": float(msg_l.fidelity),
                    "expire_time_ps": int(msg_l.expire_time) if self.is_success else now_ps,
                    "swap_success": bool(self.is_success),
                }
            )
        self.update_resource_manager(self.left_memo, "RAW")
        self.update_resource_manager(self.right_memo, "RAW")


class LoggingStockSwappingA(EntanglementSwappingA):
    """Stock swapping behavior plus diagnostics logging."""

    def start(self) -> None:
        now_ps = self.owner.timeline.now()
        swap_cfg = getattr(self.owner.timeline, "swap_calibration", None)
        left_snapshot = float(self.left_memo.fidelity)
        right_snapshot = float(self.right_memo.fidelity)
        left_wait_s = max(0.0, (now_ps - self.left_memo.last_update_time) / PS_PER_S)
        right_wait_s = max(0.0, (now_ps - self.right_memo.last_update_time) / PS_PER_S)

        assert self.left_memo.entangled_memory["node_id"] == self.left_node
        assert self.right_memo.entangled_memory["node_id"] == self.right_node

        samplers = getattr(self.owner.timeline, "runtime_pair_samplers", {})
        for memo in (self.left_memo, self.right_memo):
            remote_memo = self.owner.timeline.get_entity_by_name(memo.entangled_memory["memo_id"])
            if remote_memo is not None:
                sampler = samplers.get(remote_memo.name)
                if sampler is not None:
                    sampler.sample("swap")

        if self.owner.get_generator().random() < self.success_probability():
            fidelity = self.updated_fidelity(self.left_memo.fidelity, self.right_memo.fidelity)
            self.is_success = True
            default_expire_time = min(self.left_memo.get_expire_time(), self.right_memo.get_expire_time())
            expire_time = _resolve_generated_expire_time_ps(
                timeline=self.owner.timeline,
                stage="swapped",
                now_ps=now_ps,
                default_expire_time_ps=default_expire_time,
                local_memory=self.left_memo,
                remote_memory=self.right_memo,
                left_memory=self.left_memo,
                right_memory=self.right_memo,
                swap_cfg=swap_cfg,
            )

            msg_l = EntanglementSwappingMessage(
                SwappingMsgType.SWAP_RES,
                self.left_protocol_name,
                fidelity=fidelity,
                remote_node=self.right_memo.entangled_memory["node_id"],
                remote_memo=self.right_memo.entangled_memory["memo_id"],
                expire_time=expire_time,
                meas_res=[],
            )
            msg_r = EntanglementSwappingMessage(
                SwappingMsgType.SWAP_RES,
                self.right_protocol_name,
                fidelity=fidelity,
                remote_node=self.left_memo.entangled_memory["node_id"],
                remote_memo=self.left_memo.entangled_memory["memo_id"],
                expire_time=expire_time,
                meas_res=[],
            )
        else:
            msg_l = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES, self.left_protocol_name, fidelity=0)
            msg_r = EntanglementSwappingMessage(SwappingMsgType.SWAP_RES, self.right_protocol_name, fidelity=0)

        self.owner.send_message(self.left_node, msg_l)
        self.owner.send_message(self.right_node, msg_r)
        diagnostics = getattr(self.owner.timeline, "swap_diagnostics", None)
        if diagnostics is not None:
            left_info = self.owner.resource_manager.memory_manager.get_info_by_memory(self.left_memo)
            right_info = self.owner.resource_manager.memory_manager.get_info_by_memory(self.right_memo)
            diagnostics.append(
                {
                    "time_ps": now_ps,
                    "left_start_ps": self.left_memo.generation_time,
                    "right_start_ps": self.right_memo.generation_time,
                    "left_memo_name": self.left_memo.name,
                    "right_memo_name": self.right_memo.name,
                    "left_herald_ps": left_info.entangle_time,
                    "right_herald_ps": right_info.entangle_time,
                    "left_wait_s": left_wait_s,
                    "right_wait_s": right_wait_s,
                    "max_wait_s": max(left_wait_s, right_wait_s),
                    "wait_imbalance_s": abs(left_wait_s - right_wait_s),
                    "left_remote_node": self.left_memo.entangled_memory["node_id"],
                    "right_remote_node": self.right_memo.entangled_memory["node_id"],
                    "left_expire_time_ps": int(self.left_memo.get_expire_time()),
                    "right_expire_time_ps": int(self.right_memo.get_expire_time()),
                    "left_snapshot_fidelity": left_snapshot,
                    "right_snapshot_fidelity": right_snapshot,
                    "left_refreshed_fidelity": left_snapshot,
                    "right_refreshed_fidelity": right_snapshot,
                    "swapped_fidelity": float(msg_l.fidelity),
                    "expire_time_ps": int(msg_l.expire_time) if self.is_success else now_ps,
                    "swap_success": bool(self.is_success),
                }
            )
        self.update_resource_manager(self.left_memo, "RAW")
        self.update_resource_manager(self.right_memo, "RAW")


class LoggingStockSwappingB(EntanglementSwappingB):
    """Stock swapping end protocol plus a distinct type for additive stock tracing."""


class BellDiagonalSwappingB(EntanglementSwappingB):
    """End-node swapping protocol that reinstates a Bell-diagonal state at confirmation."""

    def received_message(self, src: str, msg: "EntanglementSwappingMessage") -> None:
        from sequence.resource_management.memory_manager import MemoryInfo

        assert src == self.remote_node_name

        if msg.fidelity > 0 and self.owner.timeline.now() < msg.expire_time:
            if msg.meas_res == [1, 0]:
                self.owner.timeline.quantum_manager.run_circuit(self.z_cir, [self.memory.qstate_key])
            elif msg.meas_res == [0, 1]:
                self.owner.timeline.quantum_manager.run_circuit(self.x_cir, [self.memory.qstate_key])
            elif msg.meas_res == [1, 1]:
                self.owner.timeline.quantum_manager.run_circuit(self.x_z_cir, [self.memory.qstate_key])

            self.memory.fidelity = msg.fidelity
            self.memory.entangled_memory["node_id"] = msg.remote_node
            self.memory.entangled_memory["memo_id"] = msg.remote_memo

            remote_memory = self.owner.timeline.get_entity_by_name(msg.remote_memo)
            if remote_memory is not None:
                remaining = max(0.0, 1 - msg.fidelity)
                state = [msg.fidelity, remaining / 3, remaining / 3, remaining / 3]
                self.owner.timeline.quantum_manager.set([self.memory.qstate_key, remote_memory.qstate_key], state)
                self.memory.last_update_time = self.owner.timeline.now()
                remote_memory.last_update_time = self.owner.timeline.now()

            self.memory.update_expire_time(msg.expire_time)
            self.update_resource_manager(self.memory, MemoryInfo.ENTANGLED)
        else:
            self.update_resource_manager(self.memory, MemoryInfo.RAW)


def _es_rule_actionA_bell(memories_info: list[Any], args: dict[str, Any]) -> tuple[BellDiagonalSwappingA, list[str], list[Any], list[dict[str, Any]]]:
    protocol = BellDiagonalSwappingA(
        None,
        f"ESA.{memories_info[0].memory.name}.{memories_info[1].memory.name}",
        memories_info[0].memory,
        memories_info[1].memory,
        success_prob=args["es_succ_prob"],
        degradation=args["es_degradation"],
    )
    dsts = [info.remote_node for info in memories_info]
    req_args = [{"target_memo": memories_info[0].remote_memo}, {"target_memo": memories_info[1].remote_memo}]
    from sequence.network_management.reservation import es_req_func
    return protocol, dsts, [es_req_func, es_req_func], req_args


def _es_rule_actionB_bell(memories_info: list[Any], args: dict[str, Any]) -> tuple[BellDiagonalSwappingB, list[None], list[None], list[None]]:
    memory = memories_info[0].memory
    protocol = BellDiagonalSwappingB(None, "ESB." + memory.name, memory)
    return protocol, [None], [None], [None]


def _es_rule_actionA_stock(memories_info: list[Any], args: dict[str, Any]) -> tuple[LoggingStockSwappingA, list[str], list[Any], list[dict[str, Any]]]:
    protocol = LoggingStockSwappingA(
        None,
        f"ESA.{memories_info[0].memory.name}.{memories_info[1].memory.name}",
        memories_info[0].memory,
        memories_info[1].memory,
        success_prob=args["es_succ_prob"],
        degradation=args["es_degradation"],
    )
    dsts = [info.remote_node for info in memories_info]
    req_args = [{"target_memo": memories_info[0].remote_memo}, {"target_memo": memories_info[1].remote_memo}]
    from sequence.network_management.reservation import es_req_func
    return protocol, dsts, [es_req_func, es_req_func], req_args


def _es_rule_actionB_stock(memories_info: list[Any], args: dict[str, Any]) -> tuple[LoggingStockSwappingB, list[None], list[None], list[None]]:
    memory = memories_info[0].memory
    protocol = LoggingStockSwappingB(None, "ESB." + memory.name, memory)
    return protocol, [None], [None], [None]


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sequence.network_management.network_manager import NetworkManager
    from sequence.resource_management.rule_manager import Rule


def _reservation_imports():
    from sequence.network_management.reservation import (
        ResourceReservationProtocol,
        eg_rule_action1,
        eg_rule_action2,
        eg_rule_condition,
        ep_rule_action1,
        ep_rule_action2,
        ep_rule_condition1,
        ep_rule_condition2,
        es_rule_actionB,
        es_rule_conditionA,
        es_rule_conditionB1,
        es_rule_conditionB2,
    )
    from sequence.resource_management.rule_manager import Rule
    return {
        "ResourceReservationProtocol": ResourceReservationProtocol,
        "eg_rule_action1": eg_rule_action1,
        "eg_rule_action2": eg_rule_action2,
        "eg_rule_condition": eg_rule_condition,
        "ep_rule_action1": ep_rule_action1,
        "ep_rule_action2": ep_rule_action2,
        "ep_rule_condition1": ep_rule_condition1,
        "ep_rule_condition2": ep_rule_condition2,
        "es_rule_actionB": es_rule_actionB,
        "es_rule_conditionA": es_rule_conditionA,
        "es_rule_conditionB1": es_rule_conditionB1,
        "es_rule_conditionB2": es_rule_conditionB2,
        "Rule": Rule,
    }

def _bell_create_rules(self, path: list[str], reservation: Any) -> list["Rule"]:
        imports = _reservation_imports()
        eg_rule_action1 = imports["eg_rule_action1"]
        eg_rule_action2 = imports["eg_rule_action2"]
        eg_rule_condition = imports["eg_rule_condition"]
        ep_rule_action1 = imports["ep_rule_action1"]
        ep_rule_action2 = imports["ep_rule_action2"]
        ep_rule_condition1 = imports["ep_rule_condition1"]
        ep_rule_condition2 = imports["ep_rule_condition2"]
        es_rule_conditionA = imports["es_rule_conditionA"]
        es_rule_conditionB1 = imports["es_rule_conditionB1"]
        es_rule_conditionB2 = imports["es_rule_conditionB2"]
        Rule = imports["Rule"]
        rules = []
        memory_indices = []
        for card in self.timecards:
            if reservation in card.reservations:
                memory_indices.append(card.memory_index)

        index = path.index(self.owner.name)

        if index > 0:
            condition_args = {"memory_indices": memory_indices[:reservation.memory_size]}
            action_args = {"mid": self.owner.map_to_middle_node[path[index - 1]], "path": path, "index": index}
            rules.append(Rule(10, eg_rule_action1, eg_rule_condition, action_args, condition_args))

        if index < len(path) - 1:
            condition_args = {"memory_indices": memory_indices[:reservation.memory_size] if index == 0 else memory_indices[reservation.memory_size:]}
            action_args = {"mid": self.owner.map_to_middle_node[path[index + 1]], "path": path, "index": index, "name": self.owner.name, "reservation": reservation}
            rules.append(Rule(10, eg_rule_action2, eg_rule_condition, action_args, condition_args))

        if index > 0:
            condition_args = {"memory_indices": memory_indices[:reservation.memory_size], "reservation": reservation, "purification_mode": self.purification_mode}
            rules.append(Rule(10, ep_rule_action1, ep_rule_condition1, {}, condition_args))

        if index < len(path) - 1:
            condition_args = {
                "memory_indices": memory_indices if index == 0 else memory_indices[reservation.memory_size:],
                "fidelity": reservation.fidelity,
                "purification_mode": self.purification_mode,
            }
            rules.append(Rule(10, ep_rule_action2, ep_rule_condition2, {}, condition_args))

        if index == 0:
            condition_args = {"memory_indices": memory_indices, "target_remote": path[-1], "fidelity": reservation.fidelity}
            rules.append(Rule(10, _es_rule_actionB_bell, es_rule_conditionB1, {}, condition_args))
        elif index == len(path) - 1:
            condition_args = {"memory_indices": memory_indices, "target_remote": path[0], "fidelity": reservation.fidelity}
            rules.append(Rule(10, _es_rule_actionB_bell, es_rule_conditionB1, {}, condition_args))
        else:
            _path = path[:]
            while _path.index(self.owner.name) % 2 == 0:
                new_path = []
                for i, node in enumerate(_path):
                    if i % 2 == 0 or i == len(_path) - 1:
                        new_path.append(node)
                _path = new_path
            _index = _path.index(self.owner.name)
            left, right = _path[_index - 1], _path[_index + 1]
            condition_args = {"memory_indices": memory_indices, "left": left, "right": right, "fidelity": reservation.fidelity}
            action_args = {"es_succ_prob": self.es_succ_prob, "es_degradation": self.es_degradation}
            rules.append(Rule(10, _es_rule_actionA_bell, es_rule_conditionA, action_args, condition_args))
            rules.append(Rule(10, _es_rule_actionB_bell, es_rule_conditionB2, {}, condition_args))

        for rule in rules:
            rule.set_reservation(reservation)
        return rules


def _stock_logged_create_rules(self, path: list[str], reservation: Any) -> list["Rule"]:
        imports = _reservation_imports()
        eg_rule_action1 = imports["eg_rule_action1"]
        eg_rule_action2 = imports["eg_rule_action2"]
        eg_rule_condition = imports["eg_rule_condition"]
        ep_rule_action1 = imports["ep_rule_action1"]
        ep_rule_action2 = imports["ep_rule_action2"]
        ep_rule_condition1 = imports["ep_rule_condition1"]
        ep_rule_condition2 = imports["ep_rule_condition2"]
        es_rule_conditionA = imports["es_rule_conditionA"]
        es_rule_conditionB1 = imports["es_rule_conditionB1"]
        es_rule_conditionB2 = imports["es_rule_conditionB2"]
        Rule = imports["Rule"]
        rules = []
        memory_indices = []
        for card in self.timecards:
            if reservation in card.reservations:
                memory_indices.append(card.memory_index)

        index = path.index(self.owner.name)

        if index > 0:
            condition_args = {"memory_indices": memory_indices[:reservation.memory_size]}
            action_args = {"mid": self.owner.map_to_middle_node[path[index - 1]], "path": path, "index": index}
            rules.append(Rule(10, eg_rule_action1, eg_rule_condition, action_args, condition_args))

        if index < len(path) - 1:
            condition_args = {"memory_indices": memory_indices[:reservation.memory_size] if index == 0 else memory_indices[reservation.memory_size:]}
            action_args = {"mid": self.owner.map_to_middle_node[path[index + 1]], "path": path, "index": index, "name": self.owner.name, "reservation": reservation}
            rules.append(Rule(10, eg_rule_action2, eg_rule_condition, action_args, condition_args))

        if index > 0:
            condition_args = {"memory_indices": memory_indices[:reservation.memory_size], "reservation": reservation, "purification_mode": self.purification_mode}
            rules.append(Rule(10, ep_rule_action1, ep_rule_condition1, {}, condition_args))

        if index < len(path) - 1:
            condition_args = {
                "memory_indices": memory_indices if index == 0 else memory_indices[reservation.memory_size:],
                "fidelity": reservation.fidelity,
                "purification_mode": self.purification_mode,
            }
            rules.append(Rule(10, ep_rule_action2, ep_rule_condition2, {}, condition_args))

        if index == 0:
            condition_args = {"memory_indices": memory_indices, "target_remote": path[-1], "fidelity": reservation.fidelity}
            rules.append(Rule(10, _es_rule_actionB_stock, es_rule_conditionB1, {}, condition_args))
        elif index == len(path) - 1:
            condition_args = {"memory_indices": memory_indices, "target_remote": path[0], "fidelity": reservation.fidelity}
            rules.append(Rule(10, _es_rule_actionB_stock, es_rule_conditionB1, {}, condition_args))
        else:
            _path = path[:]
            while _path.index(self.owner.name) % 2 == 0:
                new_path = []
                for i, node in enumerate(_path):
                    if i % 2 == 0 or i == len(_path) - 1:
                        new_path.append(node)
                _path = new_path
            _index = _path.index(self.owner.name)
            left, right = _path[_index - 1], _path[_index + 1]
            condition_args = {"memory_indices": memory_indices, "left": left, "right": right, "fidelity": reservation.fidelity}
            action_args = {"es_succ_prob": self.es_succ_prob, "es_degradation": self.es_degradation}
            rules.append(Rule(10, _es_rule_actionA_stock, es_rule_conditionA, action_args, condition_args))
            rules.append(Rule(10, _es_rule_actionB_stock, es_rule_conditionB2, {}, condition_args))

        for rule in rules:
            rule.set_reservation(reservation)
        return rules


_BELL_RESERVATION_PROTOCOL_CLS = None
_STOCK_LOGGED_RESERVATION_PROTOCOL_CLS = None


def _get_bell_reservation_protocol_cls():
    global _BELL_RESERVATION_PROTOCOL_CLS
    if _BELL_RESERVATION_PROTOCOL_CLS is None:
        base_cls = _reservation_imports()["ResourceReservationProtocol"]
        _BELL_RESERVATION_PROTOCOL_CLS = type(
            "BellDiagonalReservationProtocol",
            (base_cls,),
            {"create_rules": _bell_create_rules},
        )
    return _BELL_RESERVATION_PROTOCOL_CLS


def _get_stock_logged_reservation_protocol_cls():
    global _STOCK_LOGGED_RESERVATION_PROTOCOL_CLS
    if _STOCK_LOGGED_RESERVATION_PROTOCOL_CLS is None:
        base_cls = _reservation_imports()["ResourceReservationProtocol"]
        _STOCK_LOGGED_RESERVATION_PROTOCOL_CLS = type(
            "StockLoggedReservationProtocol",
            (base_cls,),
            {"create_rules": _stock_logged_create_rules},
        )
    return _STOCK_LOGGED_RESERVATION_PROTOCOL_CLS


def _new_network_manager_bell(owner: QuantumRouter, memory_array_name: str) -> "NetworkManager":
    from sequence.network_management.network_manager import NetworkManager
    from sequence.network_management.routing import StaticRoutingProtocol

    manager = NetworkManager(owner, [])
    routing = StaticRoutingProtocol(owner, owner.name + ".StaticRoutingProtocol", {})
    rsvp_cls = _get_bell_reservation_protocol_cls()
    rsvp = rsvp_cls(owner, owner.name + ".RSVP", memory_array_name)
    rsvp.set_swapping_success_rate(0.5)
    routing.upper_protocols.append(rsvp)
    rsvp.lower_protocols.append(routing)
    manager.load_stack([routing, rsvp])
    return manager


def _new_network_manager_stock(owner: QuantumRouter, memory_array_name: str) -> "NetworkManager":
    from sequence.network_management.network_manager import NetworkManager
    from sequence.network_management.routing import StaticRoutingProtocol

    manager = NetworkManager(owner, [])
    routing = StaticRoutingProtocol(owner, owner.name + ".StaticRoutingProtocol", {})
    rsvp_cls = _get_stock_logged_reservation_protocol_cls()
    rsvp = rsvp_cls(owner, owner.name + ".RSVP", memory_array_name)
    rsvp.set_swapping_success_rate(0.5)
    routing.upper_protocols.append(rsvp)
    rsvp.lower_protocols.append(routing)
    manager.load_stack([routing, rsvp])
    return manager


class ExtendedQuantumRouter(QuantumRouter):
    def init_managers(self, memo_arr_name: str):
        from sequence.resource_management.resource_manager import ResourceManager

        self.set_resource_manager(ResourceManager(self, memo_arr_name))
        self.set_network_manager(_new_network_manager_bell(self, memo_arr_name))


class StockQuantumRouter(QuantumRouter):
    def init_managers(self, memo_arr_name: str):
        from sequence.resource_management.resource_manager import ResourceManager

        self.set_resource_manager(ResourceManager(self, memo_arr_name))
        self.set_network_manager(_new_network_manager_stock(self, memo_arr_name))


class _RuleShim:
    def __init__(self):
        self.protocols: list[Any] = []


def _register_protocol(owner, protocol, memories: list[Any]) -> None:
    owner.protocols.append(protocol)
    protocol.rule = _RuleShim()
    protocol.rule.protocols.append(protocol)
    for memory in memories:
        memory.attach(protocol)


def _attach_rule_verified_protocol(owner, protocol, memories: list[Any], *, waiting: bool = False) -> None:
    """Attach a protocol the same way a rule-created protocol is attached."""

    from sequence.resource_management.memory_manager import MemoryInfo

    protocol.owner = owner
    protocol.rule = _RuleShim()
    protocol.rule.protocols.append(protocol)
    for memory in memories:
        memory.detach(memory.memory_array)
        memory.attach(protocol)
        info = owner.resource_manager.memory_manager.get_info_by_memory(memory)
        if info.state != MemoryInfo.OCCUPIED:
            owner.resource_manager.memory_manager.update(memory, MemoryInfo.OCCUPIED)
    if waiting:
        owner.resource_manager.waiting_protocols.append(protocol)


def _bell_swap_b_request_selector(protocols: list[Any], args: dict[str, Any]):
    target_memo = args["target_memo"]
    for protocol in protocols:
        if isinstance(protocol, BellDiagonalSwappingB) and protocol.memory.name == target_memo:
            return protocol
    return None


def start_rule_verified_bell_swap(
    middle_router,
    left_router,
    right_router,
    left_middle_memory,
    right_middle_memory,
    left_end_memory,
    right_end_memory,
    *,
    name_prefix: str,
    swap: SwapCalibration,
) -> BellDiagonalSwappingA:
    """Start swapping through the SeQUeNCe resource-manager request/response path."""

    swap_a = BellDiagonalSwappingA(
        middle_router,
        f"{name_prefix}.swapA",
        left_middle_memory,
        right_middle_memory,
        success_prob=swap.success_prob,
        degradation=swap.degradation,
    )
    swap_b_left = BellDiagonalSwappingB(left_router, f"{name_prefix}.swapB.left", left_end_memory)
    swap_b_right = BellDiagonalSwappingB(right_router, f"{name_prefix}.swapB.right", right_end_memory)

    _attach_rule_verified_protocol(left_router, swap_b_left, [left_end_memory], waiting=True)
    _attach_rule_verified_protocol(right_router, swap_b_right, [right_end_memory], waiting=True)
    _attach_rule_verified_protocol(middle_router, swap_a, [left_middle_memory, right_middle_memory], waiting=False)

    middle_router.resource_manager.send_request(
        swap_a,
        left_router.name,
        _bell_swap_b_request_selector,
        {"target_memo": left_end_memory.name},
    )
    middle_router.resource_manager.send_request(
        swap_a,
        right_router.name,
        _bell_swap_b_request_selector,
        {"target_memo": right_end_memory.name},
    )
    return swap_a


def _shortest_path_distances(nodes: list, edges_m: list[tuple[str, str, float]]) -> dict[str, dict[str, float]]:
    names = [node.name for node in nodes]
    distances = {src: {dst: float("inf") for dst in names} for src in names}
    for name in names:
        distances[name][name] = 0.0

    for src, dst, distance_m in edges_m:
        distances[src][dst] = min(distances[src][dst], distance_m)
        distances[dst][src] = min(distances[dst][src], distance_m)

    for mid in names:
        for src in names:
            for dst in names:
                through_mid = distances[src][mid] + distances[mid][dst]
                if through_mid < distances[src][dst]:
                    distances[src][dst] = through_mid

    return distances


def _classical_channel_delay_ps(distance_m: float) -> int:
    return round(distance_m / SPEED_OF_LIGHT + 10 * MICROSECOND)


def _reservation_start_time_ps(pair_distances_m: dict[str, dict[str, float]], delay_ps: int | None = None) -> int:
    if delay_ps is not None:
        return max(int(1e9), 5 * delay_ps)

    max_delay_ps = 0
    for src_distances in pair_distances_m.values():
        for distance_m in src_distances.values():
            if distance_m != float("inf"):
                max_delay_ps = max(max_delay_ps, _classical_channel_delay_ps(distance_m))
    return max(int(1e9), 5 * max_delay_ps)


def _connect_classical_fully(nodes: list, pair_distances_m: dict[str, dict[str, float]], delay_ps: int | None = None) -> None:
    for src in nodes:
        for dst in nodes:
            if src is dst:
                continue
            distance_m = pair_distances_m[src.name][dst.name]
            if delay_ps is None:
                channel = ClassicalChannel(f"cc.{src.name}.{dst.name}", src.timeline, distance_m)
            else:
                channel = ClassicalChannel(f"cc.{src.name}.{dst.name}", src.timeline, distance_m, delay=delay_ps)
            channel.set_ends(src, dst.name)


def _make_quantum_channel(name: str, timeline: Timeline, sender, receiver_name: str, cfg: ElementaryLinkConfig) -> HeraldedFiberChannel:
    spec = cfg.fiber_spec or FiberSpec(
        wavelength_m=1550e-9,
        quantum_wavelength_nm=1550.0,
    )
    channel = HeraldedFiberChannel(
        name,
        timeline,
        attenuation=cfg.attenuation_db_per_m,
        distance=cfg.bsm_segment_distance_m,
        polarization_fidelity=1.0,
        frequency=cfg.memory_frequency_hz,
        spec=spec,
        sections=cfg.fiber_sections,
    )
    channel.set_ends(sender, receiver_name)
    return channel


def _install_generation_pair(router_a: QuantumRouter, router_b: QuantumRouter, bsm_node: BSMNode, cfg: ElementaryLinkConfig, memory_index_a: int, memory_index_b: int, name_prefix: str):
    memory_a = router_a.get_components_by_type("MemoryArray")[0][memory_index_a]
    memory_b = router_b.get_components_by_type("MemoryArray")[0][memory_index_b]
    _apply_link_memory_noise(memory_a, cfg.local_coherence_time_s or cfg.coherence_time_s, cfg.memory_decoherence_errors, cfg.cutoff_ratio)
    _apply_link_memory_noise(memory_b, cfg.remote_coherence_time_s or cfg.coherence_time_s, cfg.memory_decoherence_errors, cfg.cutoff_ratio)

    proto_a = EntanglementGenerationA.create(
        router_a,
        f"{name_prefix}.{router_a.name}",
        bsm_node.name,
        router_b.name,
        memory_a,
        raw_fidelity=cfg.raw_fidelity,
        raw_epr_errors=cfg.raw_epr_errors,
    )
    proto_b = EntanglementGenerationA.create(
        router_b,
        f"{name_prefix}.{router_b.name}",
        bsm_node.name,
        router_a.name,
        memory_b,
        raw_fidelity=cfg.raw_fidelity,
        raw_epr_errors=cfg.raw_epr_errors,
    )

    _register_protocol(router_a, proto_a, [memory_a])
    _register_protocol(router_b, proto_b, [memory_b])
    if bsm_node.eg not in bsm_node.protocols:
        bsm_node.protocols.append(bsm_node.eg)

    proto_a.set_others(proto_b.name, router_b.name, [memory_b.name])
    proto_b.set_others(proto_a.name, router_a.name, [memory_a.name])

    return proto_a, proto_b, memory_a, memory_b


def _apply_link_memory_noise(memory, coherence_time_s: float, decoherence_errors: list[float] | None, cutoff_ratio: float) -> None:
    """Apply per-link-end memory noise parameters before protocol use."""

    memory.coherence_time = coherence_time_s
    memory.decoherence_rate = 1 / coherence_time_s if coherence_time_s > 0 else 0
    memory.decoherence_errors = decoherence_errors
    memory.cutoff_ratio = cutoff_ratio


def _pair_state_from_memory(memory) -> list[float]:
    return list(memory.timeline.quantum_manager.get(memory.qstate_key).state)


def _pair_fidelity_from_memory(memory) -> float:
    return float(_pair_state_from_memory(memory)[0])


def _memory_fidelity(memory) -> float:
    try:
        return _virtual_pair_fidelity(memory, memory.timeline.now())
    except Exception:
        return float(memory.fidelity)


def _use_pair_level_single_heralded() -> None:
    EntanglementGenerationA.set_global_type(PAIR_LEVEL_SINGLE_HERALDED)
    EntanglementGenerationB.set_global_type(PAIR_LEVEL_SINGLE_HERALDED)


def _configure_generated_lifetime_controls(
    timeline: Timeline,
    *,
    link_calibrations_by_middle: dict[str, ElementaryLinkConfig] | None = None,
    swap_calibration: SwapCalibration | None = None,
    policy: EntanglementLifetimePolicy | None = None,
) -> None:
    timeline.link_calibrations_by_middle = dict(link_calibrations_by_middle or {})
    timeline.swap_calibration = swap_calibration
    timeline.entanglement_lifetime_policy = policy


def _virtual_pair_fidelity(memory, sample_time_ps: int) -> float:
    quantum_manager = memory.timeline.quantum_manager
    if memory.qstate_key not in quantum_manager.states:
        return float(memory.fidelity)

    state_now = np.array(quantum_manager.get(memory.qstate_key).state, dtype=float)
    return _propagate_bds_fidelity(state_now, memory.decoherence_errors, memory.decoherence_rate, int(memory.last_update_time), sample_time_ps)


def _event_sample(sampler: RuntimePairSampler, label: str, target_time_ps: int | None = None) -> dict[str, Any] | None:
    samples = sampler.event_samples.get(label, [])
    if not samples:
        return None
    if target_time_ps is None:
        return samples[-1]
    return min(samples, key=lambda sample: abs(int(sample["time_ps"]) - int(target_time_ps)))


def _propagate_bds_fidelity(state_now: np.ndarray, decoherence_errors: list[float] | None, decoherence_rate: float, last_update_time_ps: int | None, sample_time_ps: int) -> float:
    if decoherence_errors is None or last_update_time_ps is None or last_update_time_ps <= 0 or sample_time_ps <= last_update_time_ps:
        return float(state_now[0])

    idle_s = (sample_time_ps - last_update_time_ps) * 1e-12
    x_rate = decoherence_rate * decoherence_errors[0]
    y_rate = decoherence_rate * decoherence_errors[1]
    z_rate = decoherence_rate * decoherence_errors[2]

    exp_xy = np.exp(-2 * (x_rate + y_rate) * idle_s)
    exp_xz = np.exp(-2 * (x_rate + z_rate) * idle_s)
    exp_yz = np.exp(-2 * (y_rate + z_rate) * idle_s)
    p_i = (1 + exp_xy + exp_xz + exp_yz) / 4
    p_x = (1 - exp_xy - exp_xz + exp_yz) / 4
    p_y = (1 - exp_xy + exp_xz - exp_yz) / 4
    p_z = (1 + exp_xy - exp_xz - exp_yz) / 4
    transform_mtx = np.array(
        [
            [p_i, p_z, p_x, p_y],
            [p_z, p_i, p_y, p_x],
            [p_x, p_y, p_i, p_z],
            [p_y, p_x, p_z, p_i],
        ]
    )
    return float((transform_mtx @ state_now)[0])


def run_two_node_trial(
    link: ElementaryLinkConfig,
    seed: int = 0,
    stop_time_s: float = 0.05,
    target_fidelity: float | None = None,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    _use_pair_level_single_heralded()

    timeline = Timeline(int(0.02 * PS_PER_S), formalism=BELL_DIAGONAL_STATE_FORMALISM)
    _configure_generated_lifetime_controls(
        timeline,
        link_calibrations_by_middle={"m12": link},
        policy=lifetime_policy,
    )
    r1 = ExtendedQuantumRouter("r1", timeline, memo_size=1, component_templates=link.component_templates(1), seed=seed)
    r2 = ExtendedQuantumRouter("r2", timeline, memo_size=1, component_templates=link.component_templates(1), seed=seed + 1)
    bsm = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=seed + 2, component_templates=link.bsm_templates())

    r1.add_bsm_node(bsm.name, r2.name)
    r2.add_bsm_node(bsm.name, r1.name)

    classical_distances = _shortest_path_distances(
        [r1, r2, bsm],
        [
            (r1.name, bsm.name, link.bsm_segment_distance_m),
            (r2.name, bsm.name, link.bsm_segment_distance_m),
        ],
    )
    _connect_classical_fully([r1, r2, bsm], classical_distances, link.classical_delay_ps)
    _make_quantum_channel("qc.r1.m12", timeline, r1, bsm.name, link)
    _make_quantum_channel("qc.r2.m12", timeline, r2, bsm.name, link)
    r1.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")
    r2.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r1")

    request_start_time_ps = _reservation_start_time_ps(classical_distances, link.classical_delay_ps)
    timeline.init()
    requested_fidelity = link.raw_fidelity if target_fidelity is None else target_fidelity
    r1.network_manager.request("r2", request_start_time_ps, int(stop_time_s * PS_PER_S), 1, requested_fidelity)
    timeline.run()

    info = r1.resource_manager.memory_manager[0]
    success = info.state == "ENTANGLED" and info.remote_node == "r2"
    latency_s = ((info.entangle_time - request_start_time_ps) / PS_PER_S) if success else stop_time_s
    fidelity = _memory_fidelity(info.memory) if success else 0.0

    return {
        "success": success,
        "latency_s": latency_s,
        "fidelity": fidelity,
        "stop_time_s": stop_time_s,
        "request_start_time_s": request_start_time_ps / PS_PER_S,
        "target_fidelity": requested_fidelity,
    }


def collect_two_node_decay_trace(
    link: ElementaryLinkConfig,
    seed: int = 0,
    generation_window_s: float = 0.05,
    sample_points: int = 60,
    horizon_factor: float = 1.1,
    max_seed_tries: int = 1,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    max_seed_tries = max(1, max_seed_tries)
    for seed_offset in range(max_seed_tries):
        current_seed = seed + seed_offset

        _use_pair_level_single_heralded()

        observation_horizon_s = link.coherence_time_s * horizon_factor
        trace_horizon_s = generation_window_s + observation_horizon_s + 1e-6
        schedule_points = max(sample_points, int(np.ceil(trace_horizon_s / max(observation_horizon_s, 1e-12))) * sample_points)
        timeline = Timeline(int(trace_horizon_s * PS_PER_S), formalism=BELL_DIAGONAL_STATE_FORMALISM)
        _configure_generated_lifetime_controls(
            timeline,
            link_calibrations_by_middle={"m12": link},
            policy=lifetime_policy,
        )
        r1 = ExtendedQuantumRouter("r1", timeline, memo_size=1, component_templates=link.component_templates(1), seed=current_seed)
        r2 = ExtendedQuantumRouter("r2", timeline, memo_size=1, component_templates=link.component_templates(1), seed=current_seed + 1)
        bsm = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=current_seed + 2, component_templates=link.bsm_templates())
        sampled_memory = r1.get_components_by_type("MemoryArray")[0][0]
        timeline.runtime_pair_samplers = {
            sampled_memory.name: RuntimePairSampler(sampled_memory, observation_horizon_s, schedule_points)
        }

        r1.add_bsm_node(bsm.name, r2.name)
        r2.add_bsm_node(bsm.name, r1.name)

        classical_distances = _shortest_path_distances(
            [r1, r2, bsm],
            [
                (r1.name, bsm.name, link.bsm_segment_distance_m),
                (r2.name, bsm.name, link.bsm_segment_distance_m),
            ],
        )
        _connect_classical_fully([r1, r2, bsm], classical_distances, link.classical_delay_ps)
        _make_quantum_channel("qc.r1.m12", timeline, r1, bsm.name, link)
        _make_quantum_channel("qc.r2.m12", timeline, r2, bsm.name, link)
        r1.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")
        r2.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r1")

        request_start_time_ps = _reservation_start_time_ps(classical_distances, link.classical_delay_ps)
        trace_end_ps = request_start_time_ps + int(round((generation_window_s + link.coherence_time_s * horizon_factor) * PS_PER_S))
        timeline.runtime_pair_samplers[sampled_memory.name].schedule_absolute(request_start_time_ps, trace_end_ps)
        trace_app = TraceRequestApp(r1)
        timeline.init()
        trace_app.start("r2", request_start_time_ps, request_start_time_ps + int(generation_window_s * PS_PER_S), 1, link.raw_fidelity)
        timeline.run()

        success = trace_app.first_delivery_time_ps is not None and trace_app.first_delivery_memory is not None
        if not success:
            continue

        delivery = trace_app.deliveries[0]
        memory = delivery["memory"]
        entangle_time_ps = delivery["time_ps"]
        memory_start_ps = delivery["generation_time_ps"] if delivery["generation_time_ps"] is not None and delivery["generation_time_ps"] >= 0 else entangle_time_ps
        expire_time_ps = int(delivery.get("expire_time_ps", memory_start_ps + int(round(link.cutoff_ratio * link.coherence_time_s * PS_PER_S))))
        cutoff_elapsed_s = max(0.0, (expire_time_ps - memory_start_ps) / PS_PER_S)
        coherence_elapsed_s = max(link.coherence_time_s, cutoff_elapsed_s)
        horizon_elapsed_s = max(coherence_elapsed_s * horizon_factor, coherence_elapsed_s + 1e-12)
        sampler = timeline.runtime_pair_samplers[sampled_memory.name]
        if not sampler.sample_times_ps:
            continue
        herald_sample = _event_sample(sampler, "herald", entangle_time_ps) or _event_sample(sampler, "confirm", entangle_time_ps)

        end_ps = memory_start_ps + int(round(horizon_elapsed_s * PS_PER_S))
        elapsed_out: list[float] = []
        fidelity_out: list[float] = []
        memory_fidelity_out: list[float] = []
        for sample_time_ps, pair_fidelity, memory_fidelity in zip(
            sampler.sample_times_ps,
            sampler.pair_fidelities,
            sampler.memory_fidelities,
        ):
            if sample_time_ps < memory_start_ps or sample_time_ps > end_ps:
                continue
            elapsed_out.append((sample_time_ps - memory_start_ps) / PS_PER_S)
            if sample_time_ps >= expire_time_ps:
                fidelity_out.append(0.0)
                memory_fidelity_out.append(0.0)
            else:
                fidelity_out.append(float(pair_fidelity))
                memory_fidelity_out.append(float(memory_fidelity))

        return {
            "success": True,
            "seed_used": current_seed,
            "tries_used": seed_offset + 1,
            "entangle_time_s": entangle_time_ps / PS_PER_S,
            "memory_start_time_s": memory_start_ps / PS_PER_S,
            "cutoff_time_s": cutoff_elapsed_s,
            "coherence_time_s": link.coherence_time_s,
            "effective_cutoff_ratio": link.cutoff_ratio,
            "elapsed_times_s": elapsed_out,
            "fidelities": fidelity_out,
            "memory_fidelities": memory_fidelity_out,
            "herald_pair_fidelity": float(herald_sample["pair_fidelity"]) if herald_sample is not None else float("nan"),
            "herald_memory_fidelity": float(herald_sample["memory_fidelity"]) if herald_sample is not None else float("nan"),
        }

    return {
        "success": False,
        "seed_used": float("nan"),
        "tries_used": max_seed_tries,
        "entangle_time_s": float("nan"),
        "memory_start_time_s": float("nan"),
        "cutoff_time_s": float("nan"),
        "coherence_time_s": link.coherence_time_s,
        "effective_cutoff_ratio": link.cutoff_ratio,
        "elapsed_times_s": [],
        "fidelities": [],
        "memory_fidelities": [],
        "herald_pair_fidelity": float("nan"),
        "herald_memory_fidelity": float("nan"),
    }


def run_three_node_swap_trial(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    seed: int = 0,
    stop_time_s: float = 0.08,
    target_fidelity: float = 0.5,
    fidelity_aware_swapping: bool = True,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    _use_pair_level_single_heralded()

    timeline = Timeline(int(stop_time_s * PS_PER_S), formalism=BELL_DIAGONAL_STATE_FORMALISM)
    timeline.swap_diagnostics = []
    _configure_generated_lifetime_controls(
        timeline,
        link_calibrations_by_middle={"m12": left, "m23": right},
        swap_calibration=swap,
        policy=lifetime_policy,
    )
    router_cls = ExtendedQuantumRouter if fidelity_aware_swapping else StockQuantumRouter
    r1 = router_cls("r1", timeline, memo_size=1, component_templates=left.component_templates(1), seed=seed)
    r2 = router_cls("r2", timeline, memo_size=2, component_templates=left.component_templates(2), seed=seed + 1)
    r3 = router_cls("r3", timeline, memo_size=1, component_templates=right.component_templates(1), seed=seed + 2)
    m12 = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=seed + 3, component_templates=left.bsm_templates())
    m23 = ExtendedBSMNode("m23", timeline, ["r2", "r3"], seed=seed + 4, component_templates=right.bsm_templates())

    r1.add_bsm_node(m12.name, r2.name)
    r2.add_bsm_node(m12.name, r1.name)
    r2.add_bsm_node(m23.name, r3.name)
    r3.add_bsm_node(m23.name, r2.name)

    classical_delay_override = min(left.classical_delay_ps, right.classical_delay_ps) if (
        left.classical_delay_ps is not None and right.classical_delay_ps is not None
    ) else None
    classical_distances = _shortest_path_distances(
        [r1, r2, r3, m12, m23],
        [
            (r1.name, m12.name, left.bsm_segment_distance_m),
            (r2.name, m12.name, left.bsm_segment_distance_m),
            (r2.name, m23.name, right.bsm_segment_distance_m),
            (r3.name, m23.name, right.bsm_segment_distance_m),
        ],
    )
    _connect_classical_fully([r1, r2, r3, m12, m23], classical_distances, classical_delay_override)
    _make_quantum_channel("qc.r1.m12", timeline, r1, m12.name, left)
    _make_quantum_channel("qc.r2.m12", timeline, r2, m12.name, left)
    _make_quantum_channel("qc.r2.m23", timeline, r2, m23.name, right)
    _make_quantum_channel("qc.r3.m23", timeline, r3, m23.name, right)

    r1.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")
    r1.network_manager.protocol_stack[0].add_forwarding_rule("r3", "r2")
    r2.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r1")
    r2.network_manager.protocol_stack[0].add_forwarding_rule("r3", "r3")
    r3.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r2")
    r3.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")

    for router in (r1, r2, r3):
        router.network_manager.protocol_stack[1].set_swapping_success_rate(swap.success_prob)
        router.network_manager.protocol_stack[1].set_swapping_degradation(swap.degradation)

    request_start_time_ps = _reservation_start_time_ps(classical_distances, classical_delay_override)
    timeline.init()
    r1.network_manager.request("r3", request_start_time_ps, int(stop_time_s * PS_PER_S), 1, target_fidelity)
    timeline.run()

    info = r1.resource_manager.memory_manager[0]
    success = info.state == "ENTANGLED" and info.remote_node == "r3"
    latency_s = ((info.entangle_time - request_start_time_ps) / PS_PER_S) if success else stop_time_s
    fidelity = _memory_fidelity(info.memory) if success else 0.0

    return {
        "success": success,
        "latency_s": latency_s,
        "fidelity": fidelity,
        "stop_time_s": stop_time_s,
        "request_start_time_s": request_start_time_ps / PS_PER_S,
        "end_to_end_confirm_time_s": (info.entangle_time / PS_PER_S) if success else float("nan"),
        "swap_diagnostics": list(timeline.swap_diagnostics),
    }


def _summarize_run(trials: list[dict[str, Any]]) -> dict[str, float]:
    successes = sum(1 for trial in trials if trial["success"])
    total_time = sum(trial["latency_s"] if trial["success"] else trial["stop_time_s"] for trial in trials)
    fidelities = [trial["fidelity"] for trial in trials if trial["success"]]
    latencies = [trial["latency_s"] for trial in trials if trial["success"]]
    return {
        "success_probability": successes / len(trials),
        "throughput_hz": successes / total_time if total_time > 0 else 0.0,
        "fidelity_mean": mean(fidelities) if fidelities else float("nan"),
        "latency_mean_s": mean(latencies) if latencies else float("nan"),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if not isnan(value)]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return mean(finite), stdev(finite)


def summarize_experiment(per_run: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in per_run[0]:
        mu, sigma = _mean_std([run[key] for run in per_run])
        summary[key] = {"mean": mu, "std": sigma}
    return summary


def run_two_node_experiment(
    link: ElementaryLinkConfig,
    num_runs: int = 8,
    attempts_per_run: int = 50,
    base_seed: int = 100,
    stop_time_s: float = 0.05,
    target_fidelity: float | None = None,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    per_run = []
    raw_trials = []
    for run_idx in range(num_runs):
        trials = [
            run_two_node_trial(
                link,
                seed=base_seed + run_idx * 10_000 + attempt_idx,
                stop_time_s=stop_time_s,
                target_fidelity=target_fidelity,
                lifetime_policy=lifetime_policy,
            )
            for attempt_idx in range(attempts_per_run)
        ]
        raw_trials.append(trials)
        per_run.append(_summarize_run(trials))
    return {
        "config": asdict(link),
        "target_fidelity": link.raw_fidelity if target_fidelity is None else target_fidelity,
        "per_run": per_run,
        "raw_trials": raw_trials,
        "summary": summarize_experiment(per_run),
    }


def run_three_node_experiment(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    num_runs: int = 8,
    attempts_per_run: int = 50,
    base_seed: int = 500,
    stop_time_s: float = 0.08,
    target_fidelity: float = 0.5,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    per_run = []
    raw_trials = []
    for run_idx in range(num_runs):
        trials = [
            run_three_node_swap_trial(
                left,
                right,
                swap,
                seed=base_seed + run_idx * 10_000 + attempt_idx,
                stop_time_s=stop_time_s,
                target_fidelity=target_fidelity,
                lifetime_policy=lifetime_policy,
            )
            for attempt_idx in range(attempts_per_run)
        ]
        raw_trials.append(trials)
        per_run.append(_summarize_run(trials))
    return {
        "config": {
            "left": asdict(left),
            "right": asdict(right),
            "swap": asdict(swap),
        },
        "per_run": per_run,
        "raw_trials": raw_trials,
        "summary": summarize_experiment(per_run),
    }


def run_three_node_continuous_trial(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    seed: int = 0,
    horizon_s: float = 1.0,
    target_fidelity: float = 0.5,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
    rate_sample_period_s: float | None = None,
) -> dict[str, Any]:
    _use_pair_level_single_heralded()

    timeline = Timeline(int((horizon_s + 0.05) * PS_PER_S), formalism=BELL_DIAGONAL_STATE_FORMALISM)
    timeline.swap_diagnostics = []
    _configure_generated_lifetime_controls(
        timeline,
        link_calibrations_by_middle={"m12": left, "m23": right},
        swap_calibration=swap,
        policy=lifetime_policy,
    )
    r1 = ExtendedQuantumRouter("r1", timeline, memo_size=1, component_templates=left.component_templates(1), seed=seed)
    r2 = ExtendedQuantumRouter("r2", timeline, memo_size=2, component_templates=left.component_templates(2), seed=seed + 1)
    r3 = ExtendedQuantumRouter("r3", timeline, memo_size=1, component_templates=right.component_templates(1), seed=seed + 2)
    m12 = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=seed + 3, component_templates=left.bsm_templates())
    m23 = ExtendedBSMNode("m23", timeline, ["r2", "r3"], seed=seed + 4, component_templates=right.bsm_templates())

    r1.add_bsm_node(m12.name, r2.name)
    r2.add_bsm_node(m12.name, r1.name)
    r2.add_bsm_node(m23.name, r3.name)
    r3.add_bsm_node(m23.name, r2.name)

    classical_delay_override = min(left.classical_delay_ps, right.classical_delay_ps) if (
        left.classical_delay_ps is not None and right.classical_delay_ps is not None
    ) else None
    classical_distances = _shortest_path_distances(
        [r1, r2, r3, m12, m23],
        [
            (r1.name, m12.name, left.bsm_segment_distance_m),
            (r2.name, m12.name, left.bsm_segment_distance_m),
            (r2.name, m23.name, right.bsm_segment_distance_m),
            (r3.name, m23.name, right.bsm_segment_distance_m),
        ],
    )
    _connect_classical_fully([r1, r2, r3, m12, m23], classical_distances, classical_delay_override)
    _make_quantum_channel("qc.r1.m12", timeline, r1, m12.name, left)
    _make_quantum_channel("qc.r2.m12", timeline, r2, m12.name, left)
    _make_quantum_channel("qc.r2.m23", timeline, r2, m23.name, right)
    _make_quantum_channel("qc.r3.m23", timeline, r3, m23.name, right)

    r1.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")
    r1.network_manager.protocol_stack[0].add_forwarding_rule("r3", "r2")
    r2.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r1")
    r2.network_manager.protocol_stack[0].add_forwarding_rule("r3", "r3")
    r3.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r2")
    r3.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")

    for router in (r1, r2, r3):
        router.network_manager.protocol_stack[1].set_swapping_success_rate(swap.success_prob)
        router.network_manager.protocol_stack[1].set_swapping_degradation(swap.degradation)

    initiator_app = ContinuousRequestApp(r1, count_deliveries=True)
    responder_app = ContinuousRequestApp(r3, count_deliveries=False)
    request_start_time_ps = _reservation_start_time_ps(classical_distances, classical_delay_override)
    request_end_time_ps = request_start_time_ps + int(round(horizon_s * PS_PER_S))

    timeline.init()
    initiator_app.start("r3", request_start_time_ps, request_end_time_ps, 1, target_fidelity)
    timeline.run()

    throughputs_hz = initiator_app.memory_counter / horizon_s if horizon_s > 0 else 0.0
    fidelity_mean = mean(initiator_app.delivery_fidelities) if initiator_app.delivery_fidelities else float("nan")
    latency_mean_s = mean(initiator_app.delivery_intervals_s) if initiator_app.delivery_intervals_s else float("nan")
    result = {
        "reservation_approved": initiator_app.approved,
        "pair_count": initiator_app.memory_counter,
        "throughput_hz": throughputs_hz,
        "fidelity_mean": fidelity_mean,
        "latency_mean_s": latency_mean_s,
        "delivery_times_s": list(initiator_app.delivery_times_s),
        "delivery_fidelities": list(initiator_app.delivery_fidelities),
        "delivery_intervals_s": list(initiator_app.delivery_intervals_s),
        "horizon_s": horizon_s,
        "request_start_time_s": request_start_time_ps / PS_PER_S,
        "request_end_time_s": request_end_time_ps / PS_PER_S,
        "swap_diagnostics": list(timeline.swap_diagnostics),
    }
    if rate_sample_period_s is not None:
        result["rate_trace"] = initiator_app.cumulative_rate_trace(horizon_s, rate_sample_period_s)
    return result


def run_two_node_continuous_trial(
    link: ElementaryLinkConfig,
    seed: int = 0,
    horizon_s: float = 1.0,
    target_fidelity: float | None = None,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
    rate_sample_period_s: float | None = None,
) -> dict[str, Any]:
    _use_pair_level_single_heralded()

    timeline = Timeline(int((horizon_s + 0.05) * PS_PER_S), formalism=BELL_DIAGONAL_STATE_FORMALISM)
    _configure_generated_lifetime_controls(
        timeline,
        link_calibrations_by_middle={"m12": link},
        policy=lifetime_policy,
    )
    r1 = ExtendedQuantumRouter("r1", timeline, memo_size=1, component_templates=link.component_templates(1), seed=seed)
    r2 = ExtendedQuantumRouter("r2", timeline, memo_size=1, component_templates=link.component_templates(1), seed=seed + 1)
    bsm = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=seed + 2, component_templates=link.bsm_templates())

    r1.add_bsm_node(bsm.name, r2.name)
    r2.add_bsm_node(bsm.name, r1.name)

    classical_distances = _shortest_path_distances(
        [r1, r2, bsm],
        [
            (r1.name, bsm.name, link.bsm_segment_distance_m),
            (r2.name, bsm.name, link.bsm_segment_distance_m),
        ],
    )
    _connect_classical_fully([r1, r2, bsm], classical_distances, link.classical_delay_ps)
    _make_quantum_channel("qc.r1.m12", timeline, r1, bsm.name, link)
    _make_quantum_channel("qc.r2.m12", timeline, r2, bsm.name, link)
    r1.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")
    r2.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r1")

    initiator_app = ContinuousRequestApp(r1, count_deliveries=True)
    responder_app = ContinuousRequestApp(r2, count_deliveries=False)
    request_start_time_ps = _reservation_start_time_ps(classical_distances, link.classical_delay_ps)
    request_end_time_ps = request_start_time_ps + int(round(horizon_s * PS_PER_S))
    requested_fidelity = link.raw_fidelity if target_fidelity is None else target_fidelity

    timeline.init()
    initiator_app.start("r2", request_start_time_ps, request_end_time_ps, 1, requested_fidelity)
    timeline.run()

    throughput_hz = initiator_app.memory_counter / horizon_s if horizon_s > 0 else 0.0
    fidelity_mean = mean(initiator_app.delivery_fidelities) if initiator_app.delivery_fidelities else float("nan")
    latency_mean_s = mean(initiator_app.delivery_intervals_s) if initiator_app.delivery_intervals_s else float("nan")
    result = {
        "reservation_approved": initiator_app.approved,
        "pair_count": initiator_app.memory_counter,
        "throughput_hz": throughput_hz,
        "fidelity_mean": fidelity_mean,
        "latency_mean_s": latency_mean_s,
        "delivery_times_s": list(initiator_app.delivery_times_s),
        "delivery_fidelities": list(initiator_app.delivery_fidelities),
        "delivery_intervals_s": list(initiator_app.delivery_intervals_s),
        "horizon_s": horizon_s,
        "request_start_time_s": request_start_time_ps / PS_PER_S,
        "request_end_time_s": request_end_time_ps / PS_PER_S,
        "target_fidelity": requested_fidelity,
    }
    if rate_sample_period_s is not None:
        result["rate_trace"] = initiator_app.cumulative_rate_trace(horizon_s, rate_sample_period_s)
    return result


def run_two_node_continuous_experiment(
    link: ElementaryLinkConfig,
    num_runs: int = 6,
    base_seed: int = 24_000,
    horizon_s: float = 1.0,
    target_fidelity: float | None = None,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
    rate_sample_period_s: float | None = None,
) -> dict[str, Any]:
    per_run = []
    raw_trials = []
    for run_idx in range(num_runs):
        trial = run_two_node_continuous_trial(
            link,
            seed=base_seed + run_idx,
            horizon_s=horizon_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
            rate_sample_period_s=rate_sample_period_s,
        )
        raw_trials.append(trial)
        per_run.append(_summarize_continuous_run([trial]))
    return {
        "config": asdict(link),
        "target_fidelity": link.raw_fidelity if target_fidelity is None else target_fidelity,
        "horizon_s": horizon_s,
        "num_runs": num_runs,
        "per_run": per_run,
        "raw_trials": raw_trials,
        "rate_trace_summary": _average_rate_traces(raw_trials),
        "summary": summarize_experiment(per_run),
    }


def _summarize_continuous_run(trials: list[dict[str, Any]]) -> dict[str, float]:
    total_pairs = sum(trial["pair_count"] for trial in trials)
    total_horizon_s = sum(trial["horizon_s"] for trial in trials)
    fidelities = [fidelity for trial in trials for fidelity in trial["delivery_fidelities"]]
    latencies = [latency for trial in trials for latency in trial["delivery_intervals_s"]]
    return {
        "throughput_hz": total_pairs / total_horizon_s if total_horizon_s > 0 else 0.0,
        "fidelity_mean": mean(fidelities) if fidelities else float("nan"),
        "latency_mean_s": mean(latencies) if latencies else float("nan"),
        "pair_count_mean": mean([trial["pair_count"] for trial in trials]) if trials else float("nan"),
    }


def _average_rate_traces(trials: list[dict[str, Any]]) -> dict[str, list[float]]:
    traces = [trial.get("rate_trace") for trial in trials if trial.get("rate_trace")]
    if not traces:
        return {"times_s": [], "mean_rates_hz": [], "std_rates_hz": [], "mean_pair_counts": []}
    times = traces[0]["times_s"]
    mean_rates = []
    std_rates = []
    mean_counts = []
    for idx in range(len(times)):
        rates = [trace["rates_hz"][idx] for trace in traces if len(trace["rates_hz"]) > idx]
        counts = [trace["pair_counts"][idx] for trace in traces if len(trace["pair_counts"]) > idx]
        rate_mu, rate_sigma = _mean_std(rates)
        count_mu, _ = _mean_std(counts)
        mean_rates.append(rate_mu)
        std_rates.append(rate_sigma)
        mean_counts.append(count_mu)
    return {
        "times_s": times,
        "mean_rates_hz": mean_rates,
        "std_rates_hz": std_rates,
        "mean_pair_counts": mean_counts,
    }


def run_three_node_continuous_experiment(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    num_runs: int = 6,
    base_seed: int = 20_000,
    horizon_s: float = 1.0,
    target_fidelity: float = 0.5,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
    rate_sample_period_s: float | None = None,
) -> dict[str, Any]:
    per_run = []
    raw_trials = []
    for run_idx in range(num_runs):
        trial = run_three_node_continuous_trial(
            left,
            right,
            swap,
            seed=base_seed + run_idx,
            horizon_s=horizon_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
            rate_sample_period_s=rate_sample_period_s,
        )
        raw_trials.append(trial)
        per_run.append(_summarize_continuous_run([trial]))
    return {
        "config": {
            "left": asdict(left),
            "right": asdict(right),
            "swap": asdict(swap),
        },
        "target_fidelity": target_fidelity,
        "horizon_s": horizon_s,
        "num_runs": num_runs,
        "per_run": per_run,
        "raw_trials": raw_trials,
        "rate_trace_summary": _average_rate_traces(raw_trials),
        "summary": summarize_experiment(per_run),
    }


def collect_three_node_rate_over_time(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    num_runs: int = 8,
    base_seed: int = 90_000,
    horizon_s: float = 10.0,
    sample_period_s: float = 1.0,
    target_fidelity: float = 0.5,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    return run_three_node_continuous_experiment(
        left,
        right,
        swap,
        num_runs=num_runs,
        base_seed=base_seed,
        horizon_s=horizon_s,
        target_fidelity=target_fidelity,
        lifetime_policy=lifetime_policy,
        rate_sample_period_s=sample_period_s,
    )


def collect_three_node_swap_waiting_histogram(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    num_runs: int = 8,
    base_seed: int = 95_000,
    horizon_s: float = 10.0,
    target_fidelity: float = 0.5,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    out = run_three_node_continuous_experiment(
        left,
        right,
        swap,
        num_runs=num_runs,
        base_seed=base_seed,
        horizon_s=horizon_s,
        target_fidelity=target_fidelity,
        lifetime_policy=lifetime_policy,
    )
    left_waits = []
    right_waits = []
    for trial in out["raw_trials"]:
        for diag in trial.get("swap_diagnostics", []):
            if diag.get("swap_success"):
                left_waits.append(float(diag["left_wait_s"]))
                right_waits.append(float(diag["right_wait_s"]))
    out["left_wait_times_s"] = left_waits
    out["right_wait_times_s"] = right_waits
    return out


def collect_swap_waiting_points(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    num_runs: int = 6,
    attempts_per_run: int = 30,
    base_seed: int = 7_000,
    stop_time_s: float = 0.08,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for run_idx in range(num_runs):
        for attempt_idx in range(attempts_per_run):
            trial = run_three_node_swap_trial(
                left,
                right,
                swap,
                seed=base_seed + run_idx * 10_000 + attempt_idx,
                stop_time_s=stop_time_s,
                target_fidelity=0.5,
                lifetime_policy=lifetime_policy,
            )
            for diag in trial.get("swap_diagnostics", []):
                if diag["swap_success"] and trial["success"]:
                    point = dict(diag)
                    point["trial_success"] = True
                    point["final_end_to_end_fidelity"] = trial["fidelity"]
                    point["latency_s"] = trial["latency_s"]
                    points.append(point)
    return points


def collect_three_node_swap_stage_sample(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    seed: int = 0,
    stop_time_s: float = 0.08,
    target_fidelity: float = 0.5,
    max_seed_tries: int = 20,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    max_seed_tries = max(1, max_seed_tries)
    for seed_offset in range(max_seed_tries):
        current_seed = seed + seed_offset
        trial = run_three_node_swap_trial(
            left,
            right,
            swap,
            seed=current_seed,
            stop_time_s=stop_time_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
        )
        if not trial["success"] or not trial.get("swap_diagnostics"):
            continue

        successful_diags = [diag for diag in trial["swap_diagnostics"] if diag["swap_success"]]
        diag = successful_diags[-1] if successful_diags else trial["swap_diagnostics"][-1]
        tracked_side = "left" if diag.get("left_remote_node") == "r1" else "right"
        waiting_link_raw_fidelity = left.raw_fidelity if tracked_side == "left" else right.raw_fidelity
        waiting_link_snapshot_fidelity = diag["left_snapshot_fidelity"] if tracked_side == "left" else diag["right_snapshot_fidelity"]
        waiting_link_refreshed_fidelity = diag["left_refreshed_fidelity"] if tracked_side == "left" else diag["right_refreshed_fidelity"]
        stale_end_to_end_fidelity = diag["left_snapshot_fidelity"] * diag["right_snapshot_fidelity"] * swap.degradation
        waiting_link_start_s = (diag["left_start_ps"] / PS_PER_S) if tracked_side == "left" else (diag["right_start_ps"] / PS_PER_S)
        waiting_link_herald_s = (diag["left_herald_ps"] / PS_PER_S) if tracked_side == "left" else (diag["right_herald_ps"] / PS_PER_S)
        swap_time_s = diag["time_ps"] / PS_PER_S
        end_to_end_confirm_s = trial["end_to_end_confirm_time_s"]
        stage_times_s = [
            0.0,
            waiting_link_herald_s - waiting_link_start_s,
            swap_time_s - waiting_link_start_s,
            end_to_end_confirm_s - waiting_link_start_s,
        ]

        return {
            "success": True,
            "seed_used": current_seed,
            "tries_used": seed_offset + 1,
            "tracked_side": tracked_side,
            "tracked_wait_s": diag["left_wait_s"] if tracked_side == "left" else diag["right_wait_s"],
            "other_wait_s": diag["right_wait_s"] if tracked_side == "left" else diag["left_wait_s"],
            "swap_time_s": diag["time_ps"] / PS_PER_S,
            "max_wait_s": diag["max_wait_s"],
            "wait_imbalance_s": diag["wait_imbalance_s"],
            "waiting_link_raw_fidelity": waiting_link_raw_fidelity,
            "waiting_link_snapshot_fidelity": waiting_link_snapshot_fidelity,
            "waiting_link_refreshed_fidelity": waiting_link_refreshed_fidelity,
            "left_snapshot_fidelity": diag["left_snapshot_fidelity"],
            "right_snapshot_fidelity": diag["right_snapshot_fidelity"],
            "left_refreshed_fidelity": diag["left_refreshed_fidelity"],
            "right_refreshed_fidelity": diag["right_refreshed_fidelity"],
            "waiting_link_start_s": waiting_link_start_s,
            "waiting_link_herald_s": waiting_link_herald_s,
            "swap_time_s": swap_time_s,
            "end_to_end_confirm_time_s": end_to_end_confirm_s,
            "stage_times_s": stage_times_s,
            "stale_end_to_end_fidelity": stale_end_to_end_fidelity,
            "refreshed_end_to_end_fidelity": trial["fidelity"],
        }

    return {
        "success": False,
        "seed_used": float("nan"),
        "tries_used": max_seed_tries,
    }


def collect_three_node_swap_time_trace(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    swap: SwapCalibration,
    seed: int = 0,
    stop_time_s: float = 0.08,
    target_fidelity: float = 0.5,
    sample_points: int = 100,
    max_seed_tries: int = 20,
    fidelity_aware_swapping: bool = True,
    lifetime_policy: EntanglementLifetimePolicy | None = None,
) -> dict[str, Any]:
    max_seed_tries = max(1, max_seed_tries)
    for seed_offset in range(max_seed_tries):
        current_seed = seed + seed_offset

        _use_pair_level_single_heralded()

        observation_horizon_s = max(left.coherence_time_s, right.coherence_time_s) * 1.15
        trace_horizon_s = stop_time_s + observation_horizon_s + 1e-6
        schedule_points = max(sample_points, int(np.ceil(trace_horizon_s / max(observation_horizon_s, 1e-12))) * sample_points)
        timeline = Timeline(int(trace_horizon_s * PS_PER_S), formalism=BELL_DIAGONAL_STATE_FORMALISM)
        timeline.swap_diagnostics = []
        _configure_generated_lifetime_controls(
            timeline,
            link_calibrations_by_middle={"m12": left, "m23": right},
            swap_calibration=swap,
            policy=lifetime_policy,
        )
        router_cls = ExtendedQuantumRouter if fidelity_aware_swapping else StockQuantumRouter
        r1 = router_cls("r1", timeline, memo_size=1, component_templates=left.component_templates(1), seed=current_seed)
        r2 = router_cls("r2", timeline, memo_size=2, component_templates=left.component_templates(2), seed=current_seed + 1)
        r3 = router_cls("r3", timeline, memo_size=1, component_templates=right.component_templates(1), seed=current_seed + 2)
        m12 = ExtendedBSMNode("m12", timeline, ["r1", "r2"], seed=current_seed + 3, component_templates=left.bsm_templates())
        m23 = ExtendedBSMNode("m23", timeline, ["r2", "r3"], seed=current_seed + 4, component_templates=right.bsm_templates())
        sampled_memory = r1.get_components_by_type("MemoryArray")[0][0]
        timeline.runtime_pair_samplers = {
            sampled_memory.name: RuntimePairSampler(sampled_memory, observation_horizon_s, schedule_points)
        }

        r1.add_bsm_node(m12.name, r2.name)
        r2.add_bsm_node(m12.name, r1.name)
        r2.add_bsm_node(m23.name, r3.name)
        r3.add_bsm_node(m23.name, r2.name)

        classical_delay_override = min(left.classical_delay_ps, right.classical_delay_ps) if (
            left.classical_delay_ps is not None and right.classical_delay_ps is not None
        ) else None
        classical_distances = _shortest_path_distances(
            [r1, r2, r3, m12, m23],
            [
                (r1.name, m12.name, left.bsm_segment_distance_m),
                (r2.name, m12.name, left.bsm_segment_distance_m),
                (r2.name, m23.name, right.bsm_segment_distance_m),
                (r3.name, m23.name, right.bsm_segment_distance_m),
            ],
        )
        _connect_classical_fully([r1, r2, r3, m12, m23], classical_distances, classical_delay_override)
        _make_quantum_channel("qc.r1.m12", timeline, r1, m12.name, left)
        _make_quantum_channel("qc.r2.m12", timeline, r2, m12.name, left)
        _make_quantum_channel("qc.r2.m23", timeline, r2, m23.name, right)
        _make_quantum_channel("qc.r3.m23", timeline, r3, m23.name, right)

        r1.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")
        r1.network_manager.protocol_stack[0].add_forwarding_rule("r3", "r2")
        r2.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r1")
        r2.network_manager.protocol_stack[0].add_forwarding_rule("r3", "r3")
        r3.network_manager.protocol_stack[0].add_forwarding_rule("r1", "r2")
        r3.network_manager.protocol_stack[0].add_forwarding_rule("r2", "r2")

        for router in (r1, r2, r3):
            router.network_manager.protocol_stack[1].set_swapping_success_rate(swap.success_prob)
            router.network_manager.protocol_stack[1].set_swapping_degradation(swap.degradation)

        request_start_time_ps = _reservation_start_time_ps(classical_distances, classical_delay_override)
        trace_end_ps = request_start_time_ps + int(round((stop_time_s + max(left.coherence_time_s, right.coherence_time_s) * 1.15) * PS_PER_S))
        timeline.runtime_pair_samplers[sampled_memory.name].schedule_absolute(request_start_time_ps, trace_end_ps)
        trace_app = TraceRequestApp(r1)
        timeline.init()
        trace_app.start("r3", request_start_time_ps, request_start_time_ps + int(stop_time_s * PS_PER_S), 1, target_fidelity)
        timeline.run()

        if not (trace_app.deliveries and timeline.swap_diagnostics):
            continue

        successful_diags = [diag for diag in timeline.swap_diagnostics if diag["swap_success"]]
        diag = None
        delivery = None
        for candidate_diag in reversed(successful_diags if successful_diags else timeline.swap_diagnostics):
            candidate_deliveries = [candidate for candidate in trace_app.deliveries if candidate["time_ps"] >= candidate_diag["time_ps"]]
            if not candidate_deliveries:
                continue
            if fidelity_aware_swapping:
                matched_delivery = next(
                    (
                        candidate
                        for candidate in candidate_deliveries
                        if abs(float(candidate["pair_fidelity"]) - float(candidate_diag["swapped_fidelity"])) < 1e-9
                    ),
                    None,
                )
            else:
                matched_delivery = next(
                    (
                        candidate
                        for candidate in candidate_deliveries
                        if abs(float(candidate["reported_fidelity"]) - float(candidate_diag["swapped_fidelity"])) < 1e-9
                    ),
                    None,
                )
            if matched_delivery is not None:
                diag = candidate_diag
                delivery = matched_delivery
                break

        if diag is None or delivery is None:
            continue

        tracked_side = "left" if diag.get("left_remote_node") == "r1" else "right"

        end_memory = sampled_memory

        start_ps = diag["left_start_ps"] if tracked_side == "left" else diag["right_start_ps"]
        other_start_ps = diag["right_start_ps"] if tracked_side == "left" else diag["left_start_ps"]
        herald_ps = diag["left_herald_ps"] if tracked_side == "left" else diag["right_herald_ps"]
        swap_ps = diag["time_ps"]
        confirm_ps = int(round(delivery["time_ps"]))
        tracked_link = left if tracked_side == "left" else right
        other_link = right if tracked_side == "left" else left
        pre_swap_cutoff_ps = int(diag["left_expire_time_ps"] if tracked_side == "left" else diag["right_expire_time_ps"])
        pre_swap_coherence_ps = start_ps + int(round(tracked_link.coherence_time_s * PS_PER_S))
        other_cutoff_ps = int(diag["right_expire_time_ps"] if tracked_side == "left" else diag["left_expire_time_ps"])
        other_coherence_ps = other_start_ps + int(round(other_link.coherence_time_s * PS_PER_S))
        post_swap_cutoff_ps = int(diag.get("expire_time_ps", min(pre_swap_cutoff_ps, other_cutoff_ps)))
        post_swap_coherence_ps = pre_swap_coherence_ps if pre_swap_cutoff_ps <= other_cutoff_ps else other_coherence_ps
        cutoff_ps = post_swap_cutoff_ps
        sampler = timeline.runtime_pair_samplers[sampled_memory.name]
        if not sampler.sample_times_ps:
            continue
        herald_sample = _event_sample(sampler, "herald", herald_ps)
        swap_sample = _event_sample(sampler, "swap", swap_ps)
        confirm_sample = _event_sample(sampler, "confirm", confirm_ps)

        horizon_ps = max(pre_swap_coherence_ps, post_swap_coherence_ps, cutoff_ps)
        elapsed_times_s: list[float] = []
        before_trace: list[float] = []
        after_trace: list[float] = []
        last_preconfirm_state: list[float] | None = None
        last_preconfirm_update_ps: int | None = None
        last_preswap_state: list[float] | None = None
        last_preswap_update_ps: int | None = None
        for sample_idx, (sample_ps, memory_fidelity, pair_fidelity) in enumerate(zip(
            sampler.sample_times_ps,
            sampler.memory_fidelities,
            sampler.pair_fidelities,
        )):
            if sample_ps < start_ps or sample_ps > horizon_ps:
                continue
            elapsed_s = (sample_ps - start_ps) / PS_PER_S
            elapsed_times_s.append(elapsed_s)
            if sample_ps >= cutoff_ps:
                before_trace.append(0.0)
            else:
                before_trace.append(float(memory_fidelity))
            sampled_state = sampler.pair_states[sample_idx] if sample_idx < len(sampler.pair_states) else None
            sampled_update_ps = sampler.state_update_times_ps[sample_idx] if sample_idx < len(sampler.state_update_times_ps) else None

            if sample_ps >= cutoff_ps:
                after_trace.append(0.0)
                continue

            if sample_ps < confirm_ps:
                if not fidelity_aware_swapping:
                    after_trace.append(float(memory_fidelity))
                else:
                    if sample_ps < swap_ps and sampled_state is not None:
                        last_preswap_state = sampled_state
                        last_preswap_update_ps = sampled_update_ps
                    if sample_ps >= swap_ps and last_preswap_state is not None:
                        after_trace.append(
                            _propagate_bds_fidelity(
                                np.array(last_preswap_state, dtype=float),
                                tracked_link.raw_epr_errors,
                                end_memory.decoherence_rate,
                                last_preswap_update_ps,
                                sample_ps,
                            )
                        )
                    elif sampled_state is not None:
                        last_preconfirm_state = sampled_state
                        last_preconfirm_update_ps = sampled_update_ps
                        after_trace.append(float(pair_fidelity))
                    elif last_preconfirm_state is not None:
                        after_trace.append(
                            _propagate_bds_fidelity(
                                np.array(last_preconfirm_state, dtype=float),
                                tracked_link.raw_epr_errors,
                                end_memory.decoherence_rate,
                                last_preconfirm_update_ps,
                                sample_ps,
                            )
                        )
                    else:
                        after_trace.append(float(pair_fidelity))
            else:
                if fidelity_aware_swapping:
                    if sampled_state is not None:
                        after_trace.append(float(pair_fidelity))
                    else:
                        remaining = max(0.0, 1 - float(delivery["pair_fidelity"]))
                        swapped_state = np.array([float(delivery["pair_fidelity"]), remaining / 3, remaining / 3, remaining / 3], dtype=float)
                        after_trace.append(
                            _propagate_bds_fidelity(
                                swapped_state,
                                tracked_link.raw_epr_errors,
                                end_memory.decoherence_rate,
                                confirm_ps,
                                sample_ps,
                            )
                        )
                else:
                    after_trace.append(float(delivery["reported_fidelity"]))

        stale_end_to_end = diag["left_snapshot_fidelity"] * diag["right_snapshot_fidelity"] * swap.degradation
        return {
            "success": True,
            "seed_used": current_seed,
            "tries_used": seed_offset + 1,
            "tracked_side": tracked_side,
            "tracked_wait_s": diag["left_wait_s"] if tracked_side == "left" else diag["right_wait_s"],
            "other_wait_s": diag["right_wait_s"] if tracked_side == "left" else diag["left_wait_s"],
            "elapsed_times_s": elapsed_times_s,
            "before_trace": before_trace,
            "after_trace": after_trace,
            "excitation_time_s": 0.0,
            "herald_time_s": (herald_ps - start_ps) / PS_PER_S,
            "other_herald_time_s": ((diag["right_herald_ps"] if tracked_side == "left" else diag["left_herald_ps"]) - start_ps) / PS_PER_S,
            "ready_time_s": (max(diag["left_herald_ps"], diag["right_herald_ps"]) - start_ps) / PS_PER_S,
            "swap_time_s": (swap_ps - start_ps) / PS_PER_S,
            "confirm_time_s": (confirm_ps - start_ps) / PS_PER_S,
            "protocol_pairing_delay_s": (swap_ps - max(diag["left_herald_ps"], diag["right_herald_ps"])) / PS_PER_S,
            "cutoff_time_s": (post_swap_cutoff_ps - start_ps) / PS_PER_S,
            "coherence_time_s": (post_swap_coherence_ps - start_ps) / PS_PER_S,
            "pre_swap_cutoff_time_s": (pre_swap_cutoff_ps - start_ps) / PS_PER_S,
            "pre_swap_coherence_time_s": (pre_swap_coherence_ps - start_ps) / PS_PER_S,
            "post_swap_cutoff_time_s": (post_swap_cutoff_ps - start_ps) / PS_PER_S,
            "post_swap_coherence_time_s": (post_swap_coherence_ps - start_ps) / PS_PER_S,
            "stale_end_to_end_fidelity": stale_end_to_end,
            "refreshed_end_to_end_fidelity": float(delivery["pair_fidelity"] if fidelity_aware_swapping else delivery["reported_fidelity"]),
            "left_swap_snapshot_fidelity": float(diag["left_snapshot_fidelity"]),
            "right_swap_snapshot_fidelity": float(diag["right_snapshot_fidelity"]),
            "left_swap_used_fidelity": float(diag["left_refreshed_fidelity"] if fidelity_aware_swapping else diag["left_snapshot_fidelity"]),
            "right_swap_used_fidelity": float(diag["right_refreshed_fidelity"] if fidelity_aware_swapping else diag["right_snapshot_fidelity"]),
            "swap_result_fidelity": float(diag["swapped_fidelity"]),
            "confirm_reported_fidelity": float(delivery["reported_fidelity"]),
            "confirm_pair_fidelity": float(delivery["pair_fidelity"]),
            "herald_reported_fidelity": float(herald_sample["memory_fidelity"]) if herald_sample is not None else float("nan"),
            "herald_pair_fidelity": float(herald_sample["pair_fidelity"]) if herald_sample is not None else float("nan"),
            "swap_reported_fidelity": float(swap_sample["memory_fidelity"]) if swap_sample is not None else float("nan"),
            "swap_pair_fidelity": float(swap_sample["pair_fidelity"]) if swap_sample is not None else float("nan"),
            "confirm_trace_fidelity": float(confirm_sample["pair_fidelity"]) if confirm_sample is not None else float("nan"),
            "swap_input_before_fidelity": diag["left_snapshot_fidelity"] if tracked_side == "left" else diag["right_snapshot_fidelity"],
            "swap_input_after_fidelity": float(swap_sample["pair_fidelity"]) if (swap_sample is not None and fidelity_aware_swapping) else float(swap_sample["memory_fidelity"]) if swap_sample is not None else float(np.interp((swap_ps - start_ps) / PS_PER_S, elapsed_times_s, after_trace)),
        }

    return {
        "success": False,
        "seed_used": float("nan"),
        "tries_used": max_seed_tries,
    }


def sweep_swap_waiting_imbalance(
    left: ElementaryLinkConfig,
    right: ElementaryLinkConfig,
    right_link_success_rates: list[float],
    swap: SwapCalibration,
    num_runs: int = 4,
    attempts_per_run: int = 20,
    base_seed: int = 9_000,
    stop_time_s: float = 0.08,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for idx, success_rate in enumerate(right_link_success_rates):
        right_cfg = replace(right, bsm_success_rate=success_rate)
        points = collect_swap_waiting_points(
            left,
            right_cfg,
            swap,
            num_runs=num_runs,
            attempts_per_run=attempts_per_run,
            base_seed=base_seed + idx * 1_000,
            stop_time_s=stop_time_s,
        )
        wait_values = [point["max_wait_s"] for point in points]
        refreshed_values = [
            point["left_refreshed_fidelity"] if point["left_wait_s"] >= point["right_wait_s"] else point["right_refreshed_fidelity"]
            for point in points
        ]
        end_to_end_values = [point["final_end_to_end_fidelity"] for point in points]
        wait_mu, wait_sigma = _mean_std(wait_values) if wait_values else (float("nan"), float("nan"))
        refreshed_mu, refreshed_sigma = _mean_std(refreshed_values) if refreshed_values else (float("nan"), float("nan"))
        end_mu, end_sigma = _mean_std(end_to_end_values) if end_to_end_values else (float("nan"), float("nan"))
        rows.append(
            {
                "right_bsm_success_rate": success_rate,
                "mean_wait_s": wait_mu,
                "std_wait_s": wait_sigma,
                "mean_waiting_link_fidelity": refreshed_mu,
                "std_waiting_link_fidelity": refreshed_sigma,
                "mean_end_to_end_fidelity": end_mu,
                "std_end_to_end_fidelity": end_sigma,
                "samples": float(len(points)),
            }
        )
    return rows


def metric_table(result: dict[str, Any], include_deadline_completion: bool = False) -> list[dict[str, float | str]]:
    rows = []
    for metric, stats in result["summary"].items():
        if metric == "success_probability" and not include_deadline_completion:
            continue
        rows.append({"metric": metric, "mean": stats["mean"], "std": stats["std"]})
    return rows


def sweep_two_node_raw_fidelity(link: ElementaryLinkConfig, fidelities: list[float], num_runs: int = 6, attempts_per_run: int = 30, base_seed: int = 2_000, stop_time_s: float = 0.05) -> list[dict[str, float]]:
    results = []
    for idx, fidelity in enumerate(fidelities):
        cfg = replace(link, raw_fidelity=fidelity)
        out = run_two_node_experiment(cfg, num_runs=num_runs, attempts_per_run=attempts_per_run, base_seed=base_seed + idx * 1_000, stop_time_s=stop_time_s)
        results.append(
            {
                "raw_fidelity": fidelity,
                "throughput_mean": out["summary"]["throughput_hz"]["mean"],
                "throughput_std": out["summary"]["throughput_hz"]["std"],
                "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
                "fidelity_std": out["summary"]["fidelity_mean"]["std"],
            }
        )
    return results


def sweep_three_node_swap_degradation(left: ElementaryLinkConfig, right: ElementaryLinkConfig, degradations: list[float], swap_success_prob: float = 1.0, num_runs: int = 6, attempts_per_run: int = 30, base_seed: int = 4_000, stop_time_s: float = 0.08) -> list[dict[str, float]]:
    results = []
    for idx, degradation in enumerate(degradations):
        swap = SwapCalibration(success_prob=swap_success_prob, degradation=degradation)
        out = run_three_node_experiment(left, right, swap, num_runs=num_runs, attempts_per_run=attempts_per_run, base_seed=base_seed + idx * 1_000, stop_time_s=stop_time_s)
        results.append(
            {
                "swap_degradation": degradation,
                "throughput_mean": out["summary"]["throughput_hz"]["mean"],
                "throughput_std": out["summary"]["throughput_hz"]["std"],
                "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
                "fidelity_std": out["summary"]["fidelity_mean"]["std"],
            }
        )
    return results


def sweep_two_node_target_fidelity(link: ElementaryLinkConfig, target_fidelities: list[float], num_runs: int = 6, attempts_per_run: int = 30, base_seed: int = 12_000, stop_time_s: float = 0.05, lifetime_policy: EntanglementLifetimePolicy | None = None) -> list[dict[str, float]]:
    rows = []
    for idx, target_fidelity in enumerate(target_fidelities):
        out = run_two_node_experiment(
            link,
            num_runs=num_runs,
            attempts_per_run=attempts_per_run,
            base_seed=base_seed + idx * 1_000,
            stop_time_s=stop_time_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
        )
        rows.append(
            {
                "target_fidelity": target_fidelity,
                "throughput_mean": out["summary"]["throughput_hz"]["mean"],
                "throughput_std": out["summary"]["throughput_hz"]["std"],
                "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
                "fidelity_std": out["summary"]["fidelity_mean"]["std"],
                "latency_mean_s": out["summary"]["latency_mean_s"]["mean"],
                "latency_std_s": out["summary"]["latency_mean_s"]["std"],
            }
        )
    return rows


def sweep_three_node_target_fidelity(left: ElementaryLinkConfig, right: ElementaryLinkConfig, swap: SwapCalibration, target_fidelities: list[float], num_runs: int = 6, attempts_per_run: int = 30, base_seed: int = 14_000, stop_time_s: float = 0.08, lifetime_policy: EntanglementLifetimePolicy | None = None) -> list[dict[str, float]]:
    rows = []
    for idx, target_fidelity in enumerate(target_fidelities):
        out = run_three_node_experiment(
            left,
            right,
            swap,
            num_runs=num_runs,
            attempts_per_run=attempts_per_run,
            base_seed=base_seed + idx * 1_000,
            stop_time_s=stop_time_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
        )
        rows.append(
            {
                "target_fidelity": target_fidelity,
                "throughput_mean": out["summary"]["throughput_hz"]["mean"],
                "throughput_std": out["summary"]["throughput_hz"]["std"],
                "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
                "fidelity_std": out["summary"]["fidelity_mean"]["std"],
                "latency_mean_s": out["summary"]["latency_mean_s"]["mean"],
                "latency_std_s": out["summary"]["latency_mean_s"]["std"],
            }
        )
    return rows


def sweep_two_node_cutoff_ratio(link: ElementaryLinkConfig, cutoff_ratios: list[float], target_fidelity: float, num_runs: int = 6, attempts_per_run: int = 30, base_seed: int = 16_000, stop_time_s: float = 0.05, lifetime_policy: EntanglementLifetimePolicy | None = None) -> list[dict[str, float]]:
    rows = []
    for idx, cutoff_ratio in enumerate(cutoff_ratios):
        cfg = replace(link, cutoff_ratio=cutoff_ratio)
        out = run_two_node_experiment(
            cfg,
            num_runs=num_runs,
            attempts_per_run=attempts_per_run,
            base_seed=base_seed + idx * 1_000,
            stop_time_s=stop_time_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
        )
        rows.append(
            {
                "cutoff_ratio": cutoff_ratio,
                "target_fidelity": target_fidelity,
                "throughput_mean": out["summary"]["throughput_hz"]["mean"],
                "throughput_std": out["summary"]["throughput_hz"]["std"],
                "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
                "fidelity_std": out["summary"]["fidelity_mean"]["std"],
                "latency_mean_s": out["summary"]["latency_mean_s"]["mean"],
                "latency_std_s": out["summary"]["latency_mean_s"]["std"],
            }
        )
    return rows


def sweep_three_node_cutoff_ratio(left: ElementaryLinkConfig, right: ElementaryLinkConfig, swap: SwapCalibration, cutoff_ratios: list[float], target_fidelity: float, num_runs: int = 6, attempts_per_run: int = 30, base_seed: int = 18_000, stop_time_s: float = 0.08, lifetime_policy: EntanglementLifetimePolicy | None = None) -> list[dict[str, float]]:
    rows = []
    for idx, cutoff_ratio in enumerate(cutoff_ratios):
        left_cfg = replace(left, cutoff_ratio=cutoff_ratio)
        right_cfg = replace(right, cutoff_ratio=cutoff_ratio)
        out = run_three_node_experiment(
            left_cfg,
            right_cfg,
            swap,
            num_runs=num_runs,
            attempts_per_run=attempts_per_run,
            base_seed=base_seed + idx * 1_000,
            stop_time_s=stop_time_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
        )
        rows.append(
            {
                "cutoff_ratio": cutoff_ratio,
                "target_fidelity": target_fidelity,
                "throughput_mean": out["summary"]["throughput_hz"]["mean"],
                "throughput_std": out["summary"]["throughput_hz"]["std"],
                "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
                "fidelity_std": out["summary"]["fidelity_mean"]["std"],
                "latency_mean_s": out["summary"]["latency_mean_s"]["mean"],
                "latency_std_s": out["summary"]["latency_mean_s"]["std"],
            }
        )
    return rows


def sweep_three_node_target_fidelity_continuous(left: ElementaryLinkConfig, right: ElementaryLinkConfig, swap: SwapCalibration, target_fidelities: list[float], num_runs: int = 4, base_seed: int = 30_000, horizon_s: float = 1.0, lifetime_policy: EntanglementLifetimePolicy | None = None) -> list[dict[str, float]]:
    rows = []
    for idx, target_fidelity in enumerate(target_fidelities):
        out = run_three_node_continuous_experiment(
            left,
            right,
            swap,
            num_runs=num_runs,
            base_seed=base_seed + idx * 1_000,
            horizon_s=horizon_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
        )
        rows.append(
            {
                "target_fidelity": target_fidelity,
                "throughput_mean": out["summary"]["throughput_hz"]["mean"],
                "throughput_std": out["summary"]["throughput_hz"]["std"],
                "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
                "fidelity_std": out["summary"]["fidelity_mean"]["std"],
                "latency_mean_s": out["summary"]["latency_mean_s"]["mean"],
                "latency_std_s": out["summary"]["latency_mean_s"]["std"],
                "pair_count_mean": out["summary"]["pair_count_mean"]["mean"],
                "pair_count_std": out["summary"]["pair_count_mean"]["std"],
            }
        )
    return rows


def sweep_three_node_cutoff_ratio_continuous(left: ElementaryLinkConfig, right: ElementaryLinkConfig, swap: SwapCalibration, cutoff_ratios: list[float], target_fidelity: float, num_runs: int = 4, base_seed: int = 34_000, horizon_s: float = 1.0, lifetime_policy: EntanglementLifetimePolicy | None = None) -> list[dict[str, float]]:
    rows = []
    for idx, cutoff_ratio in enumerate(cutoff_ratios):
        left_cfg = replace(left, cutoff_ratio=cutoff_ratio)
        right_cfg = replace(right, cutoff_ratio=cutoff_ratio)
        out = run_three_node_continuous_experiment(
            left_cfg,
            right_cfg,
            swap,
            num_runs=num_runs,
            base_seed=base_seed + idx * 1_000,
            horizon_s=horizon_s,
            target_fidelity=target_fidelity,
            lifetime_policy=lifetime_policy,
        )
        rows.append(
            {
                "cutoff_ratio": cutoff_ratio,
                "target_fidelity": target_fidelity,
                "throughput_mean": out["summary"]["throughput_hz"]["mean"],
                "throughput_std": out["summary"]["throughput_hz"]["std"],
                "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
                "fidelity_std": out["summary"]["fidelity_mean"]["std"],
                "latency_mean_s": out["summary"]["latency_mean_s"]["mean"],
                "latency_std_s": out["summary"]["latency_mean_s"]["std"],
                "pair_count_mean": out["summary"]["pair_count_mean"]["mean"],
                "pair_count_std": out["summary"]["pair_count_mean"]["std"],
            }
        )
    return rows
