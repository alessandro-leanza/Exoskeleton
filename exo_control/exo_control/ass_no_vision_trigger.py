#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from exo_interfaces.srv import RunTrajectory, SetAdmittanceParams


class NoVisionController(Node):
    """
    State machine senza visione:

      STATES:
        - 'stand' (iniziale)
        - 'moving_down'  -> (dopo traj_duration_s) -> 'bend'
        - 'bend'
        - 'moving_up'    -> (dopo traj_duration_s) -> 'stand'

      TRIGGERS:
        - stand: if joint_0.pos > down_trigger -> run_trajectory('down')
        - bend : if joint_0.pos < up_trigger   -> run_trajectory('up')

      ADMITTANCE K:
        - K=50 durante la traiettoria (down/up)
        - dopo 1.5 s dalla fine della traiettoria -> K=0
    """

    def __init__(self):
        super().__init__('no_vision_controller')

        # ----- Parametri -----
        self.declare_parameter('joint_name', 'joint_0')   # nome giunto; se assente, usa fallback_index
        self.declare_parameter('fallback_index', 0)

        self.declare_parameter('down_trigger', 0.15)       # [rad] per innescare 'down' da 'stand'
        self.declare_parameter('up_trigger', 1.05)        # [rad] per innescare 'up' da 'bend'

        self.declare_parameter('traj_duration_s', 2.0)    # durata stimata della traiettoria (per i timer)
        self.declare_parameter('k_traj', 30.0)            # K durante la traiettoria
        self.declare_parameter('k_idle', 0.0)             # K a riposo
        self.declare_parameter('k_reset_after_s', 1.5)    # [s] dopo fine traiettoria, metti K=0

        # (opzionali) tempi di persistenza per i trigger (0 = immediato)
        self.declare_parameter('hold_down_s', 0.0)
        self.declare_parameter('hold_up_s', 0.0)

        # ----- Carica parametri -----
        self.joint_name = str(self.get_parameter('joint_name').value)
        self.fallback_index = int(self.get_parameter('fallback_index').value)

        self.down_trigger = float(self.get_parameter('down_trigger').value)
        self.up_trigger = float(self.get_parameter('up_trigger').value)

        self.traj_duration_s = float(self.get_parameter('traj_duration_s').value)
        self.k_traj = float(self.get_parameter('k_traj').value)
        self.k_idle = float(self.get_parameter('k_idle').value)
        self.k_reset_after_s = float(self.get_parameter('k_reset_after_s').value)

        self.hold_down_s = float(self.get_parameter('hold_down_s').value)
        self.hold_up_s = float(self.get_parameter('hold_up_s').value)

        # ----- Stato runtime -----
        self.state = 'stand'            # stato iniziale
        self.jidx = None                # indice del giunto scelto
        self.above_since = None         # timer persistenza per 'down'
        self.below_since = None         # timer persistenza per 'up'

        # timer per fine traiettoria e per reset K
        self._traj_end_timer = None
        self._k_reset_timer = None

        # ----- ROS I/O -----
        self.sub_js = self.create_subscription(JointState, 'joint_states', self._on_js, 30)
        self.traj_client = self.create_client(RunTrajectory, 'run_trajectory')
        self.adm_client = self.create_client(SetAdmittanceParams, 'set_admittance_params')

        # prova a impostare K a idle all'avvio (best effort, non bloccare troppo)
        if self.adm_client.wait_for_service(timeout_sec=1.0):
            self._set_k(self.k_idle, "[INIT]")
        else:
            self.get_logger().warn("Service 'set_admittance_params' non pronto all'avvio.")

        if not self.traj_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Service 'run_trajectory' non pronto all'avvio.")

        self.get_logger().info(
            f"no_vision_controller READY | state={self.state} | joint='{self.joint_name}' "
            f"(fallback idx {self.fallback_index}) | down>{self.down_trigger:.2f} rad | up<{self.up_trigger:.2f} rad | "
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
        """Imposta K=K_traj, lancia la traiettoria 'which' e programma transizioni/timer."""
        # cancella eventuali timer vecchi
        for t in (self._traj_end_timer, self._k_reset_timer):
            if t:
                try:
                    t.cancel()
                except Exception:
                    pass

        # K alto durante la traiettoria
        self._set_k(self.k_traj, "[MOTION]")
        # avvia traiettoria
        ok = self._run_traj(which)
        if not ok:
            return

        # stato "in movimento"
        self.state = f"moving_{which}"
        self.get_logger().info(f"[STATE] → {self.state}")

        # timer: fine traiettoria
        def _on_traj_end():
            # passa allo stato successivo (bend o stand)
            self.state = next_state
            self.get_logger().info(f"[STATE] → {self.state} (traj completed)")
            # programma reset K dopo k_reset_after_s
            def _reset_k():
                self._set_k(self.k_idle, "[RESET]")
                if self._k_reset_timer:
                    try:
                        self._k_reset_timer.cancel()
                    except Exception:
                        pass
                    self._k_reset_timer = None
            self._k_reset_timer = self.create_timer(self.k_reset_after_s, _reset_k)

            # pulizia del timer di fine traiettoria
            if self._traj_end_timer:
                try:
                    self._traj_end_timer.cancel()
                except Exception:
                    pass

        self._traj_end_timer = self.create_timer(self.traj_duration_s, _on_traj_end)

    # ===== JointState callback =====
    def _on_js(self, msg: JointState):
        # risolvi indice al primo messaggio
        if self.jidx is None:
            if not msg.position:
                return
            self._resolve_joint_index(msg)

        if self.jidx >= len(msg.position):
            return
        pos = float(msg.position[self.jidx])
        now = time.monotonic()

        # --- STATE LOGIC ---
        if self.state == 'stand':
            # trigger DOWN quando supera soglia
            if pos > self.down_trigger:
                if self.hold_down_s <= 0.0:
                    self.get_logger().info(f"[TRIGGER] stand: pos {pos:.3f} > {self.down_trigger:.3f} → DOWN")
                    self._start_motion('down', next_state='bend')
                else:
                    # persistenza
                    if self.above_since is None:
                        self.above_since = now
                    elif (now - self.above_since) >= self.hold_down_s:
                        self.get_logger().info(f"[TRIGGER] stand (hold {self.hold_down_s:.2f}s): "
                                               f"pos {pos:.3f} > {self.down_trigger:.3f} → DOWN")
                        self._start_motion('down', next_state='bend')
                        self.above_since = None
            else:
                self.above_since = None

        elif self.state == 'bend':
            # trigger UP quando scende sotto soglia
            if pos < self.up_trigger:
                if self.hold_up_s <= 0.0:
                    self.get_logger().info(f"[TRIGGER] bend: pos {pos:.3f} < {self.up_trigger:.3f} → UP")
                    self._start_motion('up', next_state='stand')
                else:
                    if self.below_since is None:
                        self.below_since = now
                    elif (now - self.below_since) >= self.hold_up_s:
                        self.get_logger().info(f"[TRIGGER] bend (hold {self.hold_up_s:.2f}s): "
                                               f"pos {pos:.3f} < {self.up_trigger:.3f} → UP")
                        self._start_motion('up', next_state='stand')
                        self.below_since = None
            else:
                self.below_since = None

        else:
            # moving_* : ignora i trigger finché non arriva l'evento di fine traiettoria (timer)
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
