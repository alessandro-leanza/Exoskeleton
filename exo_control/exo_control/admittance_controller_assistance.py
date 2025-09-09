#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, Bool
from sensor_msgs.msg import JointState
from exo_interfaces.srv import SetAdmittanceParams

class AdmittanceController(Node):
    def __init__(self):
        super().__init__('admittance_controller')

        # ---------- Publishers ----------
        self.position_pub     = self.create_publisher(Float32MultiArray, 'position_cmd', 10)
        self.est_torque_pub   = self.create_publisher(Float32MultiArray, 'torque_estimated', 10)
        self.params_pub       = self.create_publisher(Float32MultiArray, 'admittance_params', 10)
        self.M_pub            = self.create_publisher(Float64, 'admittance/M', 10)
        self.C_pub            = self.create_publisher(Float64, 'admittance/C', 10)
        self.K_pub            = self.create_publisher(Float64, 'admittance/K', 10)

        # τ_meas / τ_ass (per-giunto) + contributi centrali
        self.tau_meas_pub       = self.create_publisher(Float32MultiArray, 'admittance/tau_meas', 10)
        self.tau_ass_pub        = self.create_publisher(Float32MultiArray, 'admittance/tau_ass', 10)
        self.tau_w_pub          = self.create_publisher(Float64, 'admittance/tau_w', 10)          # centrale
        self.tau_box_pub        = self.create_publisher(Float64, 'admittance/tau_box', 10)        # centrale
        self.tau_ass_total_pub  = self.create_publisher(Float64, 'admittance/tau_ass_total', 10)  # centrale (APPLICATA, rampata)

        # ---------- Subscribers & Service ----------
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.theta_ref_sub   = self.create_subscription(Float64, 'theta_ref', self.theta_ref_callback, 10)
        self.box_gate_sub    = self.create_subscription(Bool, 'perception/box_gate', self.box_gate_callback, 10)
        self.srv = self.create_service(SetAdmittanceParams, 'set_admittance_params', self.set_admittance_params_callback)

        # ---------- Admittance params ----------
        self.M = 0.5
        self.C_ratio = 0.9
        self.K = 0.0
        self.SOFT_C = 5.0
        self.C = self.SOFT_C if self.K == 0.0 else 2*self.C_ratio*math.sqrt(self.M*self.K)

        # Profili hard/soft
        self.SOFT_K = 0.0
        self.HARD_K_DEFAULT = 0.0 #30.0
        self.HARD_K_MIN = 10.0
        self.down_K = 0.0 #30.0
        self.traj_idle_s = 1.0

        # Ramp K/C (smoothstep) + hysteresis
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

        # ---------- Stato filtro & integrazione ----------
        self.dt = 0.005
        self.alpha = 0.1
        self.alpha_eff = 0.1
        self.init_positions = True
        self.last_theta_ref_time = self.get_clock().now()

        self.x = [0.0, 0.0]; self.v = [0.0, 0.0]; self.a = [0.0, 0.0]
        self.filtered_p = [0.0, 0.0]; self.filtered_v = [0.0, 0.0]; self.filtered_e = [0.0, 0.0]
        self.x_ref = [0.0, 0.0]
        self.measured_eff_zero = [0.0, 0.0]

        # ---------- Parametri modello τ_w / τ_box ----------
        self.declare_parameter('g', 9.81)
        self.declare_parameter('m_w', 10.0)          # [kg] massa equivalente tronco
        self.declare_parameter('l_w', 0.25)          # [m] braccio OW
        self.declare_parameter('m_b', 3.0)           # [kg] massa box
        self.declare_parameter('l_int', 0.35)        # [m] braccio OI
        self.declare_parameter('l_b', 0.20)          # [m] offset orizzontale box da I
        self.declare_parameter('assist_max_nm', 10.0)
        self.declare_parameter('coeff_assist', 0.5)
        self.declare_parameter('theta_r_deadzone', 0.25)
        self.declare_parameter('perception_on', True)  # abilita τ_b

        self.g = float(self.get_parameter('g').value)
        self.m_w = float(self.get_parameter('m_w').value)
        self.l_w = float(self.get_parameter('l_w').value)
        self.m_b = float(self.get_parameter('m_b').value)
        self.l_int = float(self.get_parameter('l_int').value)
        self.l_b = float(self.get_parameter('l_b').value)
        self.assist_max_nm = float(self.get_parameter('assist_max_nm').value)
        self.coeff_assist = float(self.get_parameter('coeff_assist').value)
        self.theta_r_deadzone = float(self.get_parameter('theta_r_deadzone').value)
        self.perception_on  = bool(self.get_parameter('perception_on').value)
        
        # ---------- Gating τ_ass ----------
        self.box_gate = False          # da visione/gaze/voice
        self.assist_gate = False       # ON solo durante "up"; OFF in "down" e quando idle
        self._prev_theta_ref_c = 0.0   # per stimare direzione (up/down)
        self.dir_sign = 0.0            # +1 down, -1 up (derivata di theta_ref)
        self._prev_assist_gate = False

        # ---------- Rampa τ_ass (smoothstep come K) ----------
        self.tau_time_set = 0.4
        self.tau_ass_current = 0.0
        self.tau_ass_start   = 0.0
        self.tau_ass_target  = 0.0
        self.tau_trj_active  = False
        self.tau_t0 = self.get_clock().now()
        self.tau_t1 = self.get_clock().now()
        self.tau_resched_eps = 0.3   # [Nm] variazione minima per (ri)programmare una rampa

        # ---------- Timers ----------
        self.create_timer(self.dt,  self._update_k_profile)
        self.create_timer(self.dt,  self._update_tau_ass_profile)  # <— rampa τ_ass
        self.create_timer(0.1,      self._maybe_revert_to_soft)
        self.create_timer(0.1,      self._publish_params)

        # --- STEP / LEAD DELTA SULLA REFERENCE ---
        self.declare_parameter('step_mode', True)      # abilita questa logica
        self.declare_parameter('step_delta', 0.07)     # [rad] anticipo riferimento
        self.declare_parameter('step_K', 20.0)         # rigidezza piccola durante il lead
        self.declare_parameter('step_min_speed', 0.07) # [rad/s] velocità minima per direzione valida

        self.step_mode      = bool(self.get_parameter('step_mode').value)
        self.step_delta     = float(self.get_parameter('step_delta').value)
        self.step_K         = float(self.get_parameter('step_K').value)
        self.step_min_speed = float(self.get_parameter('step_min_speed').value)

        # Stato per il lead
        self.step_dir = 0   # -1 = su, +1 = giù, 0 = fermo


        self.get_logger().info("Admittance controller initialized (with gated & ramped tau_ass).")

    # ===== Utilities K/C ramp =====
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

    # ===== Rampa τ_ass =====
    def _schedule_tau_ass(self, target: float, duration: float):
        # clamp target per sicurezza
        target = max(-self.assist_max_nm, min(self.assist_max_nm, float(target)))
        now = self.get_clock().now()
        self.tau_ass_start  = float(self.tau_ass_current)
        self.tau_ass_target = target
        self.tau_t0 = now
        self.tau_t1 = now + rclpy.duration.Duration(seconds=max(0.05, duration))
        self.tau_trj_active = True

    def _update_tau_ass_profile(self):
        if not self.tau_trj_active:
            return
        now = self.get_clock().now()
        if now >= self.tau_t1:
            self.tau_ass_current = self.tau_ass_target
            self.tau_trj_active = False
        else:
            num = (now - self.tau_t0).nanoseconds
            den = (self.tau_t1 - self.tau_t0).nanoseconds
            u = max(0.0, min(1.0, num/den)) if den > 0 else 1.0
            s = u*u*(3 - 2*u)  # smoothstep
            self.tau_ass_current = self.tau_ass_start + (self.tau_ass_target - self.tau_ass_start)*s

    # ===== Callbacks =====
    def box_gate_callback(self, msg: Bool):
        self.box_gate = bool(msg.data)

    def theta_ref_callback(self, msg: Float64):
        now = self.get_clock().now()
        was_idle = (now - self.last_theta_ref_time).nanoseconds * 1e-9 > self.traj_idle_s

        theta_ref_c = float(msg.data)   # rif. centrale (+R, -L)
        # segno direzione: cresce → +1 (down); cala → -1 (up)
        self.dir_sign = 1.0 if (theta_ref_c - self._prev_theta_ref_c) >= 0.0 else -1.0
        self._prev_theta_ref_c = theta_ref_c

        # mapping sui due giunti
        ################## self.x_ref = [theta_ref_c, -theta_ref_c]
        self.last_theta_ref_time = now

        # gating τ_ass: ON solo in "up" (dir_sign < 0), OFF in down
        self.assist_gate = (self.dir_sign < 0.0)

        # se il gate cambia, programma una rampa verso 0 oppure verso il valore corrente (verrà ri-stimato allo step successivo)
        if self.assist_gate != self._prev_assist_gate:
            # se si spegne → rampa a 0; se si accende, il prossimo joint_state programmerà il target fisico
            target = 0.0 if not self.assist_gate else self.tau_ass_current
            self._schedule_tau_ass(target, self.tau_time_set)
        self._prev_assist_gate = self.assist_gate

        # se arrivo da idle o K era basso → porta in hard (per tracking)
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

    # ===== Joint state handling =====
    def initial_soft_sync(
        self,
        x,
        max_sync_time_phase1: float = 3.0,
        sync_rate_phase1: float = 100.0,
        threshold_phase1: float = 0.01,
        v_max_phase1: float = 0.3,
        k_phase1: float = 20.0,
        max_sync_time_phase2: float = 3.0,
        sync_rate_phase2: float = 100.0,
        threshold_phase2: float = 0.01,
        alpha_phase2: float = 0.05,
        v_max_phase2: float = 0.2
    ):
        self.get_logger().info("Initial sync: Phase 0 (state init)")
        self.x = [float(x[0]), float(x[1])]

        # phase 1: right -> 0 (K=20)
        self.get_logger().info(f"Initial sync: Phase 1 (right → 0, K={k_phase1})")
        try:
            self._set_hard(K=k_phase1, duration=0.2)
        except Exception:
            self.get_logger().warn("Could not set HARD K during phase 1; continuing anyway.")

        dt1 = 1.0 / float(sync_rate_phase1)
        max_step_r = float(v_max_phase1) * dt1
        t0 = time.time()
        ok1 = False
        while (time.time() - t0) < max_sync_time_phase1:
            e_r = (0.0 - self.x[1])
            if abs(e_r) <= threshold_phase1:
                ok1 = True
                break
            step_r = max(-max_step_r, min(max_step_r, e_r))
            self.x[1] += step_r
            self.position_pub.publish(Float32MultiArray(data=self.x))
            time.sleep(dt1)

        self.position_pub.publish(Float32MultiArray(data=self.x))
        if ok1:
            self.get_logger().info(f"Phase 1 complete: xR≈0 (|e_r|≤{threshold_phase1})")
        else:
            self.get_logger().warn(f"Phase 1 timeout: xR→0 non raggiunto in {max_sync_time_phase1}s")

        # phase 2: left -> right (soft)
        try:
            self._set_soft(duration=0.1)
        except Exception:
            pass

        self.get_logger().info("Initial sync: Phase 2 (left → right, rate-limited)")
        dt2 = 1.0 / float(sync_rate_phase2)
        max_step_l = float(v_max_phase2) * dt2
        t1 = time.time()
        ok2 = False
        while (time.time() - t1) < max_sync_time_phase2:
            e = (self.x[0] - self.x[1])
            if abs(e) <= threshold_phase2:
                ok2 = True
                break
            step = -alpha_phase2 * e
            step = max(-max_step_l, min(max_step_l, step))
            self.x[0] += step
            self.position_pub.publish(Float32MultiArray(data=self.x))
            time.sleep(dt2)

        self.position_pub.publish(Float32MultiArray(data=self.x))
        if ok2:
            self.get_logger().info(f"Phase 2 complete: xL≈xR (|xL-xR|≤{threshold_phase2})")
        else:
            self.get_logger().warn(f"Phase 2 timeout: xL→xR non raggiunto in {max_sync_time_phase2}s")

        return bool(ok1 and ok2)

    def state_filter(self, msg: JointState):
        # Filtri base su pos/vel/torque
        for i in range(2):
            zeroed_effort = msg.effort[i] - self.measured_eff_zero[i]
            self.filtered_p[i] = self.alpha*msg.position[i] + (1 - self.alpha)*self.filtered_p[i]
            self.filtered_v[i] = self.alpha*msg.velocity[i] + (1 - self.alpha)*self.filtered_v[i]
            self.filtered_e[i] = self.alpha_eff*zeroed_effort + (1 - self.alpha_eff)*self.filtered_e[i]

    def _theta_central(self) -> float:
        # Stima centrale robusta (con vincolo L = -R dovrebbe valere theta ≈ right)
        return 0.5 * (self.filtered_p[0] - self.filtered_p[1])

    def _compute_tau_components(self, theta_w: float):
        # τ_w(θ) = m_w g l_w sin θ
        tau_w = self.m_w * self.g * self.l_w * math.sin(theta_w)
        # x_I(θ) = l_int sin θ  -> τ_box(θ) = m_b g (x_I + l_b)
        tau_box = self.m_b * self.g * (self.l_int * math.sin(theta_w) + self.l_b)
        return tau_w, tau_box

    def joint_state_callback(self, msg: JointState):
        if len(msg.position) < 2 or len(msg.velocity) < 2 or len(msg.effort) < 2:
            self.get_logger().warn("Joint state message incomplete.")
            return

        if self.init_positions:
            self.x = list(msg.position[:2])
            self.initial_soft_sync(
                self.x,
                v_max_phase1=0.30,
                v_max_phase2=0.20,
                threshold_phase1=0.005,
                threshold_phase2=0.01,
                alpha_phase2=0.05,
            )
            self.x_ref = list(msg.position[:2])
            self.filtered_p = list(msg.position[:2])
            self.filtered_v = list(msg.velocity[:2])
            self.filtered_e = list(msg.effort[:2])
            self.measured_eff_zero = list(msg.effort[:2])
            self.init_positions = False

        self.state_filter(msg)

        # === Gating e contributi centrali (TARGET GREZZO) ===
        theta_w = self._theta_central()
        tau_w, tau_box = self._compute_tau_components(theta_w)

        # ================== LEAD DELTA SULLA REFERENCE ==================
        if self.step_mode:
            # θ_c (centrale) e sua velocità (centrale): 0.5*(vR - vL)
            theta_c  = self._theta_central()                          # ≈ right
            theta_cv = 0.5 * (self.filtered_v[0] - self.filtered_v[1])# 0=right,1=left nel tuo setup

            # direzione dal segno della velocità
            if abs(theta_cv) > self.step_min_speed:
                self.step_dir = 1 if theta_cv > 0.0 else -1
            else:
                self.step_dir = 0

            # se c'è direzione valida, metti la ref "un po' avanti"
            if self.step_dir != 0:
                theta_ref_c = theta_c + self.step_dir * self.step_delta
                # mapping centrale → giunti
                self.x_ref = [theta_ref_c, -theta_ref_c]
                # K piccolo per inseguire morbido
                self._set_hard(K=self.step_K, duration=self.time_set)
        # ================================================================


    # # tau_ass_desired = (tau_w + (tau_box if self.box_gate else 0.0)) if self.assist_gate else 0.0
    # if self.assist_gate:
    #     #tau_raw = -(tau_w + (tau_box if self.box_gate else 0.0))  # segno per aiutare contro gravità
    #     tau_raw = tau_w + (tau_box if self.box_gate else 0.0)
    #     tau_ass_desired = self.coeff_assist * tau_raw              # ← MOLTIPLICAZIONE
    #     # clamp opzionale
    #     tau_ass_desired = max(-self.assist_max_nm, min(self.assist_max_nm, tau_ass_desired))
    # else:
    #     tau_ass_desired = 0.0

    # # clamp del target
    # tau_ass_desired = max(-self.assist_max_nm, min(self.assist_max_nm, tau_ass_desired))

        if self.assist_gate:
            # contributo fisico (stesso segno della gravità; se preferisci assistere "contro" la gravità inverti il segno qui)
            tau_raw = (tau_w + (tau_box if self.box_gate else 0.0))

            # NEW: azzera assistenza quando theta_r è vicino a 0
            theta_r_meas = self.filtered_p[0]   # indice 0 = giunto destro
            if abs(theta_r_meas) <= self.theta_r_deadzone:
                tau_ass_desired = 0.0
                # (opzionale) log molto leggero
                # self.get_logger().debug(f"Dead-zone θ_r≈0 → tau_ass=0")
            else:
                tau_ass_desired = self.coeff_assist * tau_raw

            # clamp
            tau_ass_desired = max(-self.assist_max_nm, min(self.assist_max_nm, tau_ass_desired))
        else:
            tau_ass_desired = 0.0

        # Se il target cambia in modo significativo → (ri)programma rampa di 0.4s
        if (abs(tau_ass_desired - self.tau_ass_target) > self.tau_resched_eps) or (not self.tau_trj_active and abs(tau_ass_desired - self.tau_ass_current) > self.tau_resched_eps):
            self._schedule_tau_ass(tau_ass_desired, self.tau_time_set)

        # τ_ass APPLICATA = valore rampato aggiornato dal timer
        tau_ass_applied = self.tau_ass_current

        # Per-giunto: L = -central, R = +central
        tau_ass_L = -tau_ass_applied
        tau_ass_R =  tau_ass_applied

        # Dinamica virtuale per i due giunti
        for i, tau_ass_i in enumerate([tau_ass_L, tau_ass_R]):
            self.a[i] = (tau_ass_i + self.filtered_e[i] - self.C*self.v[i] - self.K*(self.x[i] - self.x_ref[i])) / self.M
            self.v[i] += self.a[i] * self.dt
            self.x[i] += self.v[i] * self.dt

        # Vincolo L=-R (usa DOF centrale = asse destro)
        theta_ref_l = -self.x[1]
        theta_ref_r =  self.x[1]

        # Pubblicazioni
        self.position_pub.publish(Float32MultiArray(data=[theta_ref_l, theta_ref_r]))
        self.est_torque_pub.publish(Float32MultiArray(data=self.filtered_e))
        self.tau_meas_pub.publish(Float32MultiArray(data=self.filtered_e))
        self.tau_ass_pub.publish(Float32MultiArray(data=[tau_ass_L, tau_ass_R]))
        self.tau_w_pub.publish(Float64(data=float(tau_w)))
        self.tau_box_pub.publish(Float64(data=float(tau_box)))
        self.tau_ass_total_pub.publish(Float64(data=float(tau_ass_applied)))

    # ===== Policy ritorno a soft & reset gate =====
    def _maybe_revert_to_soft(self):
        now = self.get_clock().now()
        dt_idle = (now - self.last_theta_ref_time).nanoseconds * 1e-9
        # se la ref è idle → disabilita assist e rampa verso 0 se serve
        if dt_idle >= self.traj_idle_s:
            if self.assist_gate:
                self.assist_gate = False
                self._schedule_tau_ass(0.0, self.tau_time_set)
            if now >= self.hard_until:
                if self.require_still_to_soft and any(abs(v) > self.min_soft_still_vel for v in self.filtered_v):
                    return
                if not (self.mode == "soft" and not self.k_trj_active and abs(self.K_current - self.SOFT_K) < 1e-6):
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
