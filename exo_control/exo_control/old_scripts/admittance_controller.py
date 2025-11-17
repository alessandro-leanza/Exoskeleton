#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64
from sensor_msgs.msg import JointState
from exo_interfaces.srv import SetAdmittanceParams
import time
import math

class AdmittanceController(Node):
    def __init__(self):
        super().__init__('admittance_controller')

        # --- Publishers
        self.position_pub = self.create_publisher(Float32MultiArray, 'position_cmd', 10)
        self.est_torque_pub = self.create_publisher(Float32MultiArray, 'torque_estimated', 10)
        self.params_pub = self.create_publisher(Float32MultiArray, 'admittance_params', 10)
        self.M_pub = self.create_publisher(Float64, 'admittance/M', 10)
        self.C_pub = self.create_publisher(Float64, 'admittance/C', 10)
        self.K_pub = self.create_publisher(Float64, 'admittance/K', 10)

        # --- Subscribers & Services
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.theta_ref_sub   = self.create_subscription(Float64, 'theta_ref', self.theta_ref_callback, 10)
        self.srv = self.create_service(SetAdmittanceParams, 'set_admittance_params', self.set_admittance_params_callback)

        # --- Parametri ammettenza (iniziali)
        self.M = 0.5
        self.C_ratio = 0.9
        self.K = 0.0
        self.C = 5.0 if self.K == 0.0 else 2*self.C_ratio*math.sqrt(self.M*self.K)

        # --- Profili hard/soft
        self.SOFT_C = 5.0
        self.SOFT_K = 0.0
        self.HARD_K_DEFAULT = 50.0
        self.down_K = 70.0   # usato nel trigger da theta_ref

        self.HARD_K_MIN = 20.0
        self.traj_idle_s = 1.0

        # --- Stato per ramp di K/C (smoothstep) e hysteresis
        self.time_set = 0.4
        self.zeta = self.C_ratio
        self.K_current = float(self.K); self.K_start = float(self.K); self.K_target = float(self.K)
        self.C_start = float(self.C);   self.C_target = float(self.C)
        self.k_trj_active = False
        self.k_t0 = self.get_clock().now(); self.k_t1 = self.get_clock().now()

        self.HARD_HOLD_S = 0.6
        self.hard_until = self.get_clock().now()
        self.require_still_to_soft = True
        self.min_soft_still_vel = 0.10

        self.mode = "soft"  # "soft" o "hard"

        # --- Stato filtro & integrazione
        self.dt = 0.005
        self.alpha = 0.1
        self.alpha_eff = 0.1
        self.init_positions = True
        self.last_theta_ref_time = self.get_clock().now()

        self.x = [0.0, 0.0]; self.v = [0.0, 0.0]; self.a = [0.0, 0.0]
        self.filtered_p = [0.0, 0.0]; self.filtered_v = [0.0, 0.0]; self.filtered_e = [0.0, 0.0]
        self.x_ref = [0.0, 0.0]
        self.measured_eff_zero = [0.0, 0.0]

        # --- Timer periodici
        self.create_timer(self.dt,  self._update_k_profile)
        self.create_timer(0.1,      self._maybe_revert_to_soft)
        self.create_timer(0.1,      self._publish_params)

        self.get_logger().info("Admittance controller initialized (clean).")

    # ===== Utilità K/C ramp =====
    def _c_from_k(self, K: float) -> float:
        return 2*self.zeta*math.sqrt(self.M*K) if K > 0.0 else self.SOFT_C

    def _schedule_k(self, K_target: float, duration: float):
        now = self.get_clock().now()
        self.K_start   = float(self.K_current)
        self.K_target  = float(K_target)
        self.C_start   = float(self.C)
        self.C_target  = float(self._c_from_k(K_target))
        self.k_t0 = now
        self.k_t1 = now + rclpy.duration.Duration(seconds=max(0.05, duration))
        self.k_trj_active = True

    def _update_k_profile(self):
        if not self.k_trj_active:
            return
        now = self.get_clock().now()
        if now >= self.k_t1:
            self.K_current = self.K_target
            self.C = self.C_target
            self.k_trj_active = False
        else:
            num = (now - self.k_t0).nanoseconds
            den = (self.k_t1 - self.k_t0).nanoseconds
            u = max(0.0, min(1.0, num/den)) if den > 0 else 1.0
            s = u*u*(3 - 2*u)  # smoothstep
            self.K_current = self.K_start + (self.K_target - self.K_start)*s
            self.C         = self.C_start + (self.C_target - self.C_start)*s
        self.K = self.K_current

    def _set_hard(self, K=None, duration=None):
        duration = self.time_set if duration is None else duration
        if self.mode == "hard" and (K is None or abs(self.K_target - K) < 1e-6):
            return
        self.mode = "hard"
        K = self.HARD_K_DEFAULT if K is None else K
        self._schedule_k(K, duration)
        self.hard_until = self.get_clock().now() + rclpy.duration.Duration(seconds=self.HARD_HOLD_S)
        self.get_logger().info(f"[Admittance] HARD→ (ramp {duration}s) target K={K}")

    def _set_soft(self, duration=None):
        duration = self.time_set if duration is None else duration
        if self.mode == "soft" and abs(self.K_target - self.SOFT_K) < 1e-6 and not self.k_trj_active:
            return
        self.mode = "soft"
        self._schedule_k(self.SOFT_K, duration)
        self.get_logger().info(f"[Admittance] SOFT→ (ramp {duration}s) target K={self.SOFT_K}")

    # ===== Callbacks =====
    def theta_ref_callback(self, msg: Float64):
        now = self.get_clock().now()
        was_idle = (now - self.last_theta_ref_time).nanoseconds * 1e-9 > self.traj_idle_s
        self.x_ref = [msg.data, -msg.data]
        self.last_theta_ref_time = now
        if was_idle or self.K < self.HARD_K_MIN:
            self._set_hard(self.down_K, duration=self.time_set)

    def set_admittance_params_callback(self, request, response):
        target_k = float(request.k)
        if target_k <= 1e-6:
            self._set_soft(duration=self.time_set)
            msg = f"Ramping to SOFT (K=0) in {self.time_set}s"
        else:
            self._set_hard(K=target_k, duration=self.time_set)
            msg = f"Ramping to HARD (K={target_k}) in {self.time_set}s"
        response.success = True
        response.message = msg
        self.get_logger().info(msg)
        return response

    # def initial_soft_sync(self, x, max_sync_time=3.0, sync_rate=100.0, threshold=0.01, alpha=0.05):
    #     self.get_logger().info("Initial motors synchronization")
    #     dt = 1.0 / sync_rate
    #     start_time = time.time()
    #     self.x = [x[0], x[1]]
    #     while abs(self.x[0] + self.x[1]) > threshold and (time.time() - start_time) < max_sync_time:
    #         self.x[0] = (1 - alpha)*self.x[0] + alpha*(-self.x[1])
    #         self.position_pub.publish(Float32MultiArray(data=self.x))
    #         time.sleep(dt)
    #     self.x[0] = -self.x[1]
    #     self.position_pub.publish(Float32MultiArray(data=self.x))
    #     self.get_logger().info("Initial soft sync complete")

    # def initial_soft_sync(
    #     self,
    #     x,
    #     max_sync_time: float = 3.0,
    #     sync_rate: float = 100.0,
    #     threshold: float = 0.01,
    #     alpha: float = 0.05,   # fattore di correzione (mantieni basso)
    #     v_max: float = 0.2     # [rad/s] velocità massima consentita durante la sync
    # ):
    #     """
    #     Sincronizzazione iniziale muovendo SOLO l'asse sinistro,
    #     con step rate-limited per non superare v_max.
    #     """
    #     self.get_logger().info("Initial motors synchronization (left-only, rate-limited)")
    #     dt = 1.0 / float(sync_rate)
    #     start_time = time.time()

    #     # Stato iniziale del riferimento
    #     self.x = [float(x[0]), float(x[1])]

    #     # Resta 'soft' nella fase di sync per evitare strattoni (se disponibile)
    #     try:
    #         self._set_soft(duration=0.1)
    #     except Exception:
    #         pass

    #     max_step = float(v_max) * dt  # [rad] step massimo per ciclo (rispetta v_max)

    #     while abs(self.x[0] + self.x[1]) > threshold and (time.time() - start_time) < max_sync_time:
    #         e = (self.x[0] + self.x[1])            # vogliamo e -> 0 (L + R = 0)
    #         step = -alpha * e                       # muove SOLO il sinistro verso -xR

    #         # clamp dello step (limite di velocità)
    #         if step >  max_step: step =  max_step
    #         if step < -max_step: step = -max_step

    #         # aggiorna solo l'asse sinistro
    #         self.x[0] += step

    #         # pubblica i riferimenti (sinistro aggiornato, destro invariato)
    #         self.position_pub.publish(Float32MultiArray(data=self.x))

    #         time.sleep(dt)

    #     # "snap" finale clampato (niente salto grande)
    #     e = (self.x[0] + self.x[1])
    #     if abs(e) > 0.0:
    #         step = -e
    #         if step >  max_step: step =  max_step
    #         if step < -max_step: step = -max_step
    #         self.x[0] += step
    #         self.position_pub.publish(Float32MultiArray(data=self.x))

    #     if abs(self.x[0] + self.x[1]) <= threshold:
    #         self.get_logger().info(
    #             f"Initial soft sync complete (|e|={abs(self.x[0]+self.x[1]):.4f} <= {threshold})"
    #         )
    #         return True
    #     else:
    #         self.get_logger().warn(
    #             f"Initial soft sync timed out (|e|={abs(self.x[0]+self.x[1]):.4f} > {threshold})"
    #         )
    #         return False

    def initial_soft_sync(
        self,
        x,
        # --- fase 1 (destro -> 0 con K=20) ---
        max_sync_time_phase1: float = 3.0,
        sync_rate_phase1: float = 100.0,
        threshold_phase1: float = 0.01,
        v_max_phase1: float = 0.3,     # [rad/s] velocità max destro in fase 1
        k_phase1: float = 20.0,        # K desiderato in fase 1

        # --- fase 2 (sinistro -> posizione del destro, left-only rate-limited) ---
        max_sync_time_phase2: float = 3.0,
        sync_rate_phase2: float = 100.0,
        threshold_phase2: float = 0.01,
        alpha_phase2: float = 0.05,    # fattore correzione (mantenerlo basso)
        v_max_phase2: float = 0.2      # [rad/s] velocità max sinistro in fase 2
    ):
        """
        Sincronizzazione iniziale in DUE FASI.
        Fase 1: sposta SOLO l'asse destro a theta_ref=0 con K=20 (rampa breve) e v_max limitata.
        Fase 2: sposta SOLO l'asse sinistro fino a raggiungere la posizione del destro
                con la logica left-only rate-limited (nessuno snap finale).

        Args:
            x: posizioni iniziali [xL, xR]
        Returns:
            bool: True se entrambe le fasi hanno raggiunto la soglia; False se qualche fase va in timeout.
        """

        # ---------------- FASE 0: setup stato interno ----------------
        self.get_logger().info("Initial sync: Phase 0 (state init)")
        self.x = [float(x[0]), float(x[1])]

        # Allinea lo stato interno dell'admittance (evita salti quando partirà il loop)
        try:
            # se hai funzioni per allineare interno a misura, usale qui (opzionale)
            pass
        except Exception:
            pass

        # ---------------- FASE 1: destro -> 0 con K=20 ----------------
        self.get_logger().info(f"Initial sync: Phase 1 (right → 0, K={k_phase1})")

        # rampa a K=20 (breve)
        try:
            self._set_hard(K=k_phase1, duration=0.2)
        except Exception:
            self.get_logger().warn("Could not set HARD K during phase 1; continuing anyway.")

        dt1 = 1.0 / float(sync_rate_phase1)
        max_step_r = float(v_max_phase1) * dt1
        t0 = time.time()
        ok1 = False

        # Muovi SOLO l'asse destro verso 0
        while (time.time() - t0) < max_sync_time_phase1:
            e_r = (0.0 - self.x[1])      # errore del destro verso 0
            if abs(e_r) <= threshold_phase1:
                ok1 = True
                break

            # step verso zero con rate limit
            step_r = e_r
            if step_r >  max_step_r: step_r =  max_step_r
            if step_r < -max_step_r: step_r = -max_step_r

            # aggiorna solo il destro
            self.x[1] += step_r

            # pubblica: sinistro invariato, destro aggiornato
            self.position_pub.publish(Float32MultiArray(data=self.x))
            time.sleep(dt1)

        # Pubblica lo stato raggiunto alla fine della fase 1
        self.position_pub.publish(Float32MultiArray(data=self.x))
        if ok1:
            self.get_logger().info(f"Phase 1 complete: xR≈0 (|e_r|≤{threshold_phase1})")
        else:
            self.get_logger().warn(f"Phase 1 timeout: xR→0 non raggiunto in {max_sync_time_phase1}s")

        # Durante la fase 2 vogliamo essere morbidi per evitare strattoni
        try:
            self._set_soft(duration=0.1)
        except Exception:
            pass

        # ---------------- FASE 2: sinistro -> posizione del destro (left-only) ----------------
        self.get_logger().info("Initial sync: Phase 2 (left → right, rate-limited)")

        dt2 = 1.0 / float(sync_rate_phase2)
        max_step_l = float(v_max_phase2) * dt2
        t1 = time.time()
        ok2 = False

        # target sinistro = -xR se lavori in vincolo L=-R, ma qui chiedi "sinistro nella posizione del destro".
        # Nel tuo codice di sync precedente stavi usando il vincolo L = -R (e = xL + xR).
        # Qui seguiamo esattamente la tua richiesta: xL -> xR (stesso segno).
        while (time.time() - t1) < max_sync_time_phase2:
            e = (self.x[0] - self.x[1])      # errore sinistro rispetto al destro
            if abs(e) <= threshold_phase2:
                ok2 = True
                break

            # step solo sul sinistro usando la tua logica "alpha" + rate limit
            step = -alpha_phase2 * e

            # clamp per rispettare v_max_phase2
            if step >  max_step_l: step =  max_step_l
            if step < -max_step_l: step = -max_step_l

            self.x[0] += step
            self.position_pub.publish(Float32MultiArray(data=self.x))
            time.sleep(dt2)

        # Nessuno snap finale: pubblichiamo l'ultimo stato e fine
        self.position_pub.publish(Float32MultiArray(data=self.x))

        if ok2:
            self.get_logger().info(
                f"Phase 2 complete: xL≈xR (|xL-xR|≤{threshold_phase2})"
            )
        else:
            self.get_logger().warn(
                f"Phase 2 timeout: xL→xR non raggiunto in {max_sync_time_phase2}s"
            )

        return bool(ok1 and ok2)



    def state_filter(self, msg: JointState):
        for i in range(2):
            zeroed_effort = msg.effort[i] - self.measured_eff_zero[i]
            self.filtered_p[i] = self.alpha*msg.position[i] + (1 - self.alpha)*self.filtered_p[i]
            self.filtered_v[i] = self.alpha*msg.velocity[i] + (1 - self.alpha)*self.filtered_v[i]
            self.filtered_e[i] = self.alpha_eff*zeroed_effort + (1 - self.alpha_eff)*self.filtered_e[i]

    def joint_state_callback(self, msg: JointState):
        if len(msg.position) < 2 or len(msg.velocity) < 2 or len(msg.effort) < 2:
            self.get_logger().warn("Joint state message incomplete.")
            return

        if self.init_positions:
            self.x = list(msg.position[:2])
            # self.initial_soft_sync(self.x, alpha=0.05)
            self.initial_soft_sync(
                self.x,
                v_max_phase1=0.30,
                v_max_phase2=0.20,
                threshold_phase1=0.005,
                threshold_phase2=0.01,
                alpha_phase2=0.05,   # questo è il "vecchio" alpha, ma per la fase 2
            )
            self.x_ref = list(msg.position[:2])
            self.filtered_p = list(msg.position[:2])
            self.filtered_v = list(msg.velocity[:2])
            self.filtered_e = list(msg.effort[:2])
            self.measured_eff_zero = list(msg.effort[:2])
            self.init_positions = False

        self.state_filter(msg)

        # integrazione ammettenza
        for i in range(2):
            tau_ass = 0.0  # placeholder (se in futuro vuoi aggiungere feedforward)
            self.a[i] = (tau_ass + self.filtered_e[i] - self.C*self.v[i] - self.K*(self.x[i] - self.x_ref[i])) / self.M
            self.v[i] += self.a[i] * self.dt
            self.x[i] += self.v[i] * self.dt

        # vincolo L = -R
        theta_ref_l = -self.x[1]
        theta_ref_r =  self.x[1]

        self.position_pub.publish(Float32MultiArray(data=[theta_ref_l, theta_ref_r]))
        self.est_torque_pub.publish(Float32MultiArray(data=self.filtered_e))

    # ===== Policy soft =====
    def _maybe_revert_to_soft(self):
        now = self.get_clock().now()
        dt = (now - self.last_theta_ref_time).nanoseconds * 1e-9
        if dt < self.traj_idle_s:
            return
        if now < self.hard_until:
            return
        if self.require_still_to_soft and any(abs(v) > self.min_soft_still_vel for v in self.filtered_v):
            return
        if self.mode == "soft" and not self.k_trj_active and abs(self.K_current - self.SOFT_K) < 1e-6:
            return
        self._set_soft(duration=self.time_set)

    # ===== Telemetria M, C, K =====
    def _publish_params(self):
        arr = Float32MultiArray()
        arr.data = [float(self.M), float(self.C), float(self.K)]
        self.params_pub.publish(arr)
        self.M_pub.publish(Float64(data=float(self.M)))
        self.C_pub.publish(Float64(data=float(self.C)))
        self.K_pub.publish(Float64(data=float(self.K)))

def main(args=None):
    rclpy.init(args=args)
    node = AdmittanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down admittance controller.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32MultiArray, Float64
# from sensor_msgs.msg import JointState
# from exo_interfaces.srv import SetAdmittanceParams
# import copy
# import time
# import math
# import numpy as np


# class AdmittanceController(Node):
#     def __init__(self):
#         super().__init__('admittance_controller')

#         self.position_pub = self.create_publisher(Float32MultiArray, 'position_cmd', 10)
#         self.position_filt_pub = self.create_publisher(Float32MultiArray, 'position_filtered', 10)
#         self.est_torque_pub = self.create_publisher(Float32MultiArray, 'torque_estimated', 10)

#         self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
#         self.external_torque_sub = self.create_subscription(Float32MultiArray, 'external_torque', self.external_torque_callback, 10)

#         # New subscription to theta_ref topic
#         self.theta_ref_sub = self.create_subscription(Float64, 'theta_ref', self.theta_ref_callback, 10)

#         # Service to update admittance params
#         self.srv = self.create_service(SetAdmittanceParams, 'set_admittance_params', self.set_admittance_params_callback)

#         self.M = 0.5  # strong: 2.5
#         self.C = 5.0
#         self.C_ratio = 0.9
#         self.K = 0.0 # strong: 50.0 super: 70.0 low:

#         if self.K != 0.0 and self.M != 0.0:
#             self.C = 2 * self.C_ratio * (self.K * self.M) ** 0.5

#         self.SOFT_M = 0.5
#         self.SOFT_C = 5.0
#         self.SOFT_K = 0.0 

#         self.up_K = 50.0
#         self.down_K = 70.0

#         self.HARD_K_MIN = 20.0   # sopra questo K consideriamo "duro"
#         self.traj_idle_s = 1.0 

#         self.last_theta_ref_time = self.get_clock().now()
#         self.create_timer(0.1, self._maybe_revert_to_soft) 

#         self.dt = 0.005

#         # States for each joint
#         self.x = [0.0, 0.0]         # position
#         self.v = [0.0, 0.0]         # velocity
#         self.a = [0.0, 0.0]         # acceleration

#         self.filtered_p = [0.0, 0.0]         # filtered position
#         self.filtered_v = [0.0, 0.0]         # filtered velocity
#         self.filtered_e = [0.0, 0.0]         # filtered effort

#         # Desired trajectory (initially zeros)
#         self.x_ref = [0.0, 0.0]

#         # Latest measured joint states
#         self.measured_pos = [0.0, 0.0]
#         self.measured_vel = [0.0, 0.0]
#         self.measured_eff = [0.0, 0.0]
#         self.measured_eff_zero = [0.0, 0.0]

#         # External torque input (initialized to zero)
#         self.external_tau = [0.0, 0.0]

#         self.alpha = 0.1
#         self.alpha_eff = 0.1

#         self.init_positions = True

#         # Passaggio tra k diversi
#         self.HARD_K_DEFAULT = 50.0
#         self.K_current = float(self.K)
#         self.K_start = float(self.K)
#         self.K_target = float(self.K)
#         self.k_trj_active = False
#         self.k_t0 = self.get_clock().now()
#         self.k_t1 = self.get_clock().now()
#         self.zeta = self.C_ratio

#         # timer a dt per aggiornare il profilo di K
#         self.create_timer(self.dt, self._update_k_profile)

#         # (opzionali, dal messaggio precedente)
#         self.HARD_HOLD_S = 0.6
#         self.hard_until = self.get_clock().now()
#         self.require_still_to_soft = True
#         self.min_soft_still_vel = 0.10  # rad/s

#         self.mode = "soft"

#         self.C_start = float(self.C)
#         self.C_target = float(self.C)

#         self.params_pub = self.create_publisher(Float32MultiArray, 'admittance_params', 10)
#         self.M_pub = self.create_publisher(Float64, 'admittance/M', 10)
#         self.C_pub = self.create_publisher(Float64, 'admittance/C', 10)
#         self.K_pub = self.create_publisher(Float64, 'admittance/K', 10)
#         # (opzionale) stato "soft/hard"
#         # self.mode_pub = self.create_publisher(String, 'admittance/mode', 10)

#         # pubblica a 10 Hz (o quello che preferisci)
#         self.create_timer(0.1, self._publish_params)

#         self.time_set = 0.2  # default time for K ramping
        
#         self.get_logger().info("Admittance controller initialized.")

#     # def theta_ref_callback(self, msg: Float64):

#     #     self.x_ref = [msg.data, -msg.data]
#     #     self.last_theta_ref_time = self.get_clock().now()
#     #     self.get_logger().debug(f"Updated x_ref from theta_ref topic: {self.x_ref}")

#     def theta_ref_callback(self, msg: Float64):
#         now = self.get_clock().now()
#         was_idle = (now - self.last_theta_ref_time).nanoseconds * 1e-9 > self.traj_idle_s
#         self.x_ref = [msg.data, -msg.data]
#         self.last_theta_ref_time = now
#         if was_idle or self.K < self.HARD_K_MIN:
#             self._set_hard(self.down_K, duration=self.time_set)   # o self.up_K

#     def set_admittance_params_callback(self, request, response):
#         target_k = float(request.k)
#         if target_k <= 1e-6:
#             # soft con rampa
#             self._set_soft(duration=self.time_set, snap_to_ref=False)  # evita snap se vuoi zero scatto
#             msg = f"Ramping to SOFT (K=0) in {self.time_set}s"
#         else:
#             # hard con rampa al valore richiesto
#             self._set_hard(K=target_k, duration=self.time_set)
#             msg = f"Ramping to HARD (K={target_k}) in {self.time_set}s"
#         response.success = True
#         response.message = msg
#         self.get_logger().info(msg)
#         return response


#     def initial_soft_sync(self, x, max_sync_time=3.0, sync_rate=100.0, threshold=0.01, alpha=0.05):
#         self.get_logger().info("Initial motors synchronization")
#         dt = 1.0 / sync_rate
#         start_time = time.time()

#         self.x = [x[0], x[1]]  # initialize if needed

#         while abs(self.x[0] + self.x[1]) > threshold and (time.time() - start_time) < max_sync_time:
#             self.x[0] = (1 - alpha) * self.x[0] + alpha * (-self.x[1])
#             pos_msg = Float32MultiArray()
#             pos_msg.data = self.x
#             self.position_pub.publish(pos_msg)
#             time.sleep(dt)

#         # Final snap to constraint
#         self.x[0] = -self.x[1]
#         pos_msg = Float32MultiArray()
#         pos_msg.data = self.x
#         self.position_pub.publish(pos_msg)

#         self.get_logger().info("Initial soft sync complete")

#     def apply_deadband_and_saturation(self, values, deadband=0.5, limit=5.0):
#         result = []
#         for v in values:
#             if abs(v) < deadband:
#                 v_out = 0.0
#             else:
#                 if v > 0:
#                     v_out = (v - deadband)
#                 else:
#                     v_out = (v + deadband)
#             result.append(v_out)
#         return result

#     def state_filter(self, msg: JointState):
#         for i in range(2):
#             zeroed_effort = msg.effort[i] - self.measured_eff_zero[i]
#             self.filtered_p[i] = self.alpha * msg.position[i] + (1 - self.alpha) * self.filtered_p[i]
#             self.filtered_v[i] = self.alpha * msg.velocity[i] + (1 - self.alpha) * self.filtered_v[i]
#             self.filtered_e[i] = self.alpha_eff * zeroed_effort + (1 - self.alpha_eff) * self.filtered_e[i]

#     def external_torque_callback(self, msg: Float32MultiArray):
#         if len(msg.data) < 2:
#             self.get_logger().warn("External torque message incomplete.")
#             return
#         self.external_tau = list(msg.data)

#     def joint_state_callback(self, msg: JointState):
#         if len(msg.position) < 2 or len(msg.velocity) < 2:
#             self.get_logger().warn("Joint state message incomplete.")
#             return

#         if self.init_positions:
#             self.x = list(msg.position[:2])
#             self.initial_soft_sync(self.x, alpha=0.05)
#             self.x_ref = list(msg.position[:2])
#             self.filtered_p = list(msg.position[:2])
#             self.filtered_v = list(msg.velocity[:2])
#             self.filtered_e = list(msg.effort[:2])
#             self.measured_eff_zero = list(msg.effort[:2])
#             self.init_positions = False

#         self.state_filter(msg)
#         self.measured_pos = list(msg.position[:2])
#         self.measured_vel = list(msg.velocity[:2])
#         self.measured_eff = list(msg.effort[:2])

#         # Admittance dynamics integration for each joint:
#         tau_ass = [0.0, 0.0]
#         for i in range(2):
#             tau_ass[i] = self.filtered_e[i] * 0.0
#             self.a[i] = (tau_ass[i] + self.filtered_e[i] - self.C * self.v[i] - self.K * (self.x[i] - self.x_ref[i])) / self.M
#             self.v[i] += self.a[i] * self.dt
#             self.x[i] += self.v[i] * self.dt

#         theta_ref_l = copy.deepcopy(-self.x[1])
#         theta_ref_r = copy.deepcopy(self.x[1])

#         pos_msg = Float32MultiArray()
#         pos_msg.data = [theta_ref_l, theta_ref_r]
#         self.position_pub.publish(pos_msg)

#         est_torque_msg = Float32MultiArray()
#         est_torque_msg.data = self.filtered_e
#         self.est_torque_pub.publish(est_torque_msg)


#     def soft_clip(self, x, lower_limit, upper_limit, softness=0.1):
#         midpoint = 0.5 * (upper_limit + lower_limit)
#         range_half = 0.5 * (upper_limit - lower_limit)
#         return midpoint + range_half * np.tanh((x - midpoint) / (range_half * softness))
    
#     def _maybe_revert_to_soft(self):
#         now = self.get_clock().now()
#         dt = (now - self.last_theta_ref_time).nanoseconds * 1e-9
#         if dt < self.traj_idle_s:
#             return
#         if now < getattr(self, "hard_until", now):
#             return
#         if self.require_still_to_soft and any(abs(v) > self.min_soft_still_vel for v in self.filtered_v):
#             return
#         # evita retrigger: se siamo già in soft (target=SOFT_K e rampa finita), non richiamare
#         if self.mode == "soft" and not self.k_trj_active and abs(self.K_current - self.SOFT_K) < 1e-6:
#             return
#         self._set_soft(duration=self.time_set, snap_to_ref=False)


#     def _c_from_k(self, K):
#         # target smorzamento coerente: se K>0 usa 2*zeta*sqrt(M*K), se K=0 vai a SOFT_C
#         return 2 * self.zeta * math.sqrt(self.M * K) if K > 0.0 else self.SOFT_C

#     def _schedule_k(self, K_target: float, duration: float = 0.2):
#         now = self.get_clock().now()
#         self.K_start = float(self.K_current)
#         self.K_target = float(K_target)
#         self.C_start = float(self.C)                      # <--- memorizza C attuale
#         self.C_target = float(self._c_from_k(K_target))   # <--- target C coerente
#         self.k_t0 = now
#         self.k_t1 = now + rclpy.duration.Duration(seconds=max(0.05, duration))
#         self.k_trj_active = True

#     def _update_k_profile(self):
#         if not self.k_trj_active:
#             return
#         now = self.get_clock().now()
#         if now >= self.k_t1:
#             self.K_current = self.K_target
#             self.C = self.C_target                      # <--- atterra perfettamente
#             self.k_trj_active = False
#         else:
#             num = (now - self.k_t0).nanoseconds
#             den = (self.k_t1 - self.k_t0).nanoseconds
#             u = max(0.0, min(1.0, num / den)) if den > 0 else 1.0
#             s = u*u*(3 - 2*u)  # smoothstep

#             self.K_current = self.K_start + (self.K_target - self.K_start) * s
#             self.C = self.C_start + (self.C_target - self.C_start) * s  # <--- rampa C!

#         self.K = self.K_current

#     def _set_hard(self, K=None, duration=0.6):
#         if self.mode == "hard" and (K is None or abs(self.K_target - K) < 1e-6):
#             return  # già hard verso lo stesso target
#         self.mode = "hard"
#         K = self.HARD_K_DEFAULT if K is None else K
#         self._schedule_k(K, duration)
#         self.hard_until = self.get_clock().now() + rclpy.duration.Duration(
#             seconds=getattr(self, "HARD_HOLD_S", 0.0)
#         )
#         self.get_logger().info(f"[Admittance] HARD→ (ramp {duration}s) target K={K}")

#     def _set_soft(self, duration=0.6, snap_to_ref=True):
#         if self.mode == "soft" and abs(self.K_target - self.SOFT_K) < 1e-6 and not self.k_trj_active:
#             return  # già soft, niente retrigger
#         self.mode = "soft"
#         self._schedule_k(self.SOFT_K, duration)
#         if snap_to_ref:
#             self.x = self.x_ref.copy()
#             self.v = [0.0, 0.0]; self.a = [0.0, 0.0]
#         self.get_logger().info(f"[Admittance] SOFT→ (ramp {duration}s) target K={self.SOFT_K}")


#     def _publish_params(self):
#         arr = Float32MultiArray()
#         arr.data = [float(self.M), float(self.C), float(self.K)]  # ordine: M, C, K
#         self.params_pub.publish(arr)

#         self.M_pub.publish(Float64(data=float(self.M)))
#         self.C_pub.publish(Float64(data=float(self.C)))
#         self.K_pub.publish(Float64(data=float(self.K)))


# def main(args=None):
#     rclpy.init(args=args)
#     node = AdmittanceController()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("Shutting down admittance controller.")
#     finally:
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()