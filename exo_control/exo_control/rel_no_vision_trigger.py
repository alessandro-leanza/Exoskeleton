#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams


class NoVisionController(Node):
    """
    State machine con soglie RELATIVE:
      - Stato iniziale: 'stand' → salva baseline_stand = pos corrente.
      - Trigger DOWN se pos - baseline_stand >= delta_down.
      - Fine DOWN → stato 'bend' → baseline_bend = pos corrente.
      - Trigger UP se baseline_bend - pos >= delta_up.
      - K = k_traj durante le traiettorie; poi K torna a k_idle dopo k_reset_after_s.
    """

    def __init__(self):
        super().__init__('no_vision_controller')

        # ---- Parametri ----
        self.declare_parameter('joint_name', 'joint_0')       # giunto da monitorare (se msg.name presente)
        self.declare_parameter('fallback_index', 0)           # indice se i nomi non ci sono
        self.declare_parameter('delta_down', 0.04)             # [rad] spostamento avanti per trigger DOWN
        self.declare_parameter('delta_up', 0.05)               # [rad] spostamento indietro per trigger UP
        self.declare_parameter('hold_down_s', 0.0)            # [s] persistenza richiesta per DOWN (0 = immediato)
        self.declare_parameter('hold_up_s', 0.0)              # [s] persistenza richiesta per UP (0 = immediato)
        self.declare_parameter('arm_delay_s', 0.3)            # [s] tempo dopo cambio stato prima di armare i trigger

        # durata stimata delle traiettorie (deve combaciare con il service che le esegue)
        self.declare_parameter('traj_duration_s', 2.0)

        # gestione K
        self.declare_parameter('k_traj', 30.0)                # K durante le traiettorie
        self.declare_parameter('k_idle', 0.0)                 # K a riposo
        self.declare_parameter('k_reset_after_s', 1.0)        # [s] dopo fine traiettoria, torna a k_idle

        # ---- Carica ----
        self.joint_name = str(self.get_parameter('joint_name').value)
        self.fallback_index = int(self.get_parameter('fallback_index').value)
        self.delta_down = float(self.get_parameter('delta_down').value)
        self.delta_up = float(self.get_parameter('delta_up').value)
        self.hold_down_s = float(self.get_parameter('hold_down_s').value)
        self.hold_up_s = float(self.get_parameter('hold_up_s').value)
        self.arm_delay_s = float(self.get_parameter('arm_delay_s').value)
        self.traj_duration_s = float(self.get_parameter('traj_duration_s').value)
        self.k_traj = float(self.get_parameter('k_traj').value)
        self.k_idle = float(self.get_parameter('k_idle').value)
        self.k_reset_after_s = float(self.get_parameter('k_reset_after_s').value)

        # ---- Stato runtime ----
        self.state = 'stand'            # stato iniziale
        self.jidx = None
        self.last_pos = None

        self.baseline_stand = None
        self.baseline_bend = None

        self.above_since = None         # per persistenza DOWN
        self.below_since = None         # per persistenza UP
        self.arm_until = 0.0            # fino a quando non armo i trigger dopo cambio stato

        self._traj_end_timer = None
        self._k_reset_timer = None

        # ---- ROS I/O ----
        self.sub_js = self.create_subscription(JointState, 'joint_states', self._on_js, 30)
        self.traj_client = self.create_client(RunTrajectory, 'run_trajectory')
        self.adm_client = self.create_client(SetAdmittanceParams, 'set_admittance_params')

        # prova a mettere K a idle all'avvio (best-effort)
        if self.adm_client.wait_for_service(timeout_sec=1.0):
            self._set_k(self.k_idle, "[INIT]")
        else:
            self.get_logger().warn("SetAdmittanceParams non pronto all'avvio.")

        if not self.traj_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("run_trajectory non pronto all'avvio.")

        # armo i trigger tra poco (per evitare subito un down)
        self.arm_until = time.monotonic() + self.arm_delay_s

        self.get_logger().info(
            f"no_vision_controller READY | state={self.state} | joint='{self.joint_name}' "
            f"(fallback idx {self.fallback_index}) | Δdown={self.delta_down:.3f} rad | Δup={self.delta_up:.3f} rad | "
            f"K_traj={self.k_traj:.1f}, K_idle={self.k_idle:.1f}, traj={self.traj_duration_s:.2f}s, resetK+{self.k_reset_after_s:.2f}s"
        )

    # ===== Helpers =====
    def _resolve_joint_index(self, msg: JointState):
        if msg.name and self.joint_name in msg.name:
            self.jidx = msg.name.index(self.joint_name)
        else:
            self.jidx = min(self.fallback_index, len(msg.position) - 1)
        self.get_logger().info(f"[MEAS READY] Using joint index {self.jidx} "
                               f"({'name:'+self.joint_name if msg.name else 'no names'})")

    def _enter_state(self, new_state: str):
        self.state = new_state
        self.above_since = None
        self.below_since = None
        self.arm_until = time.monotonic() + self.arm_delay_s

        # aggiorna baseline allo ZERO del nuovo stato (se ho una misura)
        if self.last_pos is not None:
            if new_state == 'stand':
                self.baseline_stand = float(self.last_pos)
                self.get_logger().info(f"[STATE] → stand | baseline_stand={self.baseline_stand:.3f} rad (armed in {self.arm_delay_s:.2f}s)")
            elif new_state == 'bend':
                self.baseline_bend = float(self.last_pos)
                self.get_logger().info(f"[STATE] → bend  | baseline_bend={self.baseline_bend:.3f} rad (armed in {self.arm_delay_s:.2f}s)")
            else:
                self.get_logger().info(f"[STATE] → {new_state}")
        else:
            self.get_logger().info(f"[STATE] → {new_state} (baseline non aggiornata: nessuna misura)")

    def _set_k(self, k_value: float, prefix=""):
        if not self.adm_client.service_is_ready():
            self.get_logger().warn(f"{prefix} SetAdmittanceParams non pronto: skip K={k_value}.")
            return
        req = SetAdmittanceParams.Request()
        req.k = float(k_value)
        fut = self.adm_client.call_async(req)
        fut.add_done_callback(lambda f: self.get_logger().info(
            f"{prefix} SetAdmittanceParams(K={k_value}) → {'ok' if f.exception() is None else 'errore'}"
        ))

    def _run_traj(self, which: str):
        if not self.traj_client.service_is_ready():
            self.get_logger().warn(f"run_trajectory non pronto: comando '{which}' ignorato.")
            return False
        req = RunTrajectory.Request()
        req.trajectory_type = which
        fut = self.traj_client.call_async(req)
        fut.add_done_callback(lambda f: self.get_logger().info(
            f"RunTrajectory('{which}') → {'ok' if f.exception() is None else 'errore'}"
        ))
        return True

    def _start_motion(self, which: str, next_state: str):
        # cancella eventuali timer vecchi
        for t in (self._traj_end_timer, self._k_reset_timer):
            if t:
                try: t.cancel()
                except Exception: pass

        # K alto durante la traiettoria
        self._set_k(self.k_traj, "[MOTION]")

        # Avvia traiettoria
        if not self._run_traj(which):
            return

        moving_state = f"moving_{which}"
        self._enter_state(moving_state)  # entra nello stato "moving_*" (niente baseline qui)

        # Quando finisce la traiettoria → entra nel prossimo stato e programma reset K
        def _on_traj_end():
            self._enter_state(next_state)
            # programma reset K dopo k_reset_after_s
            def _reset_k():
                self._set_k(self.k_idle, "[RESET]")
                if self._k_reset_timer:
                    try: self._k_reset_timer.cancel()
                    except Exception: pass
                self._k_reset_timer = None
            self._k_reset_timer = self.create_timer(self.k_reset_after_s, _reset_k)

            if self._traj_end_timer:
                try: self._traj_end_timer.cancel()
                except Exception: pass

        self._traj_end_timer = self.create_timer(self.traj_duration_s, _on_traj_end)

    # ===== JointState callback =====
    def _on_js(self, msg: JointState):
        if self.jidx is None:
            if not msg.position:
                return
            self._resolve_joint_index(msg)

        if self.jidx >= len(msg.position):
            return

        pos = float(msg.position[self.jidx])
        self.last_pos = pos
        now = time.monotonic()

        # arming delay dopo cambio stato
        if now < self.arm_until:
            return

        # --- STATE LOGIC con soglie relative ---
        if self.state == 'stand':
            if self.baseline_stand is None:
                self.baseline_stand = pos  # sicurezza in caso mancasse
            delta = pos - self.baseline_stand
            # Trigger DOWN quando delta >= delta_down
            if delta >= self.delta_down:
                if self.hold_down_s <= 0.0:
                    self.get_logger().info(f"[TRIGGER] stand: Δ={delta:.3f} ≥ {self.delta_down:.3f} → DOWN")
                    self._start_motion('down', next_state='bend')
                else:
                    if self.above_since is None:
                        self.above_since = now
                    elif (now - self.above_since) >= self.hold_down_s:
                        self.get_logger().info(f"[TRIGGER] stand (hold {self.hold_down_s:.2f}s): "
                                               f"Δ={delta:.3f} ≥ {self.delta_down:.3f} → DOWN")
                        self._start_motion('down', next_state='bend')
                        self.above_since = None
            else:
                self.above_since = None

        elif self.state == 'bend':
            if self.baseline_bend is None:
                self.baseline_bend = pos
            delta = self.baseline_bend - pos
            # Trigger UP quando delta >= delta_up (cioè pos è scesa rispetto alla baseline_bend)
            if delta >= self.delta_up:
                if self.hold_up_s <= 0.0:
                    self.get_logger().info(f"[TRIGGER] bend: Δ={delta:.3f} ≥ {self.delta_up:.3f} → UP")
                    self._start_motion('up', next_state='stand')
                else:
                    if self.below_since is None:
                        self.below_since = now
                    elif (now - self.below_since) >= self.hold_up_s:
                        self.get_logger().info(f"[TRIGGER] bend (hold {self.hold_up_s:.2f}s): "
                                               f"Δ={delta:.3f} ≥ {self.delta_up:.3f} → UP")
                        self._start_motion('up', next_state='stand')
                        self.below_since = None
            else:
                self.below_since = None

        else:
            # moving_down / moving_up: i trigger sono disabilitati finché scade il timer di fine traiettoria
            pass


def main(args=None):
    rclpy.init(args=args)
    node = NoVisionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
