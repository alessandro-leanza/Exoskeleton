#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, Bool
from sensor_msgs.msg import JointState
from exo_interfaces.srv import SetAdmittanceParams

class AdmittanceController(Node):
    """
    FSM + assistenza posizionale (niente trigger di velocità).

    Ricette:
      0->1: tau_ass = 0
      1->2: tau_ass = tau_w + tau_b   (tau_b solo se box_gate=True)
      2->3: tau_ass = tau_b           (tau_b solo se box_gate=True)
      3->0: tau_ass = tau_w

    Spegnimento assist: appena raggiunto il target della transizione
    (entro +-assist_margin) E dopo che lo stato è effettivamente cambiato.
    """

    # ==== mapping stati ====
    STAND_NO_BOX   = 'stand_no_box'   # 0
    BEND_TO_PICK   = 'bend_to_pick'   # 1
    STAND_WITH_BOX = 'stand_with_box' # 2
    BEND_TO_PLACE  = 'bend_to_place'  # 3
    STATE_ID = {
        STAND_NO_BOX: 0.0,
        BEND_TO_PICK: 1.0,
        STAND_WITH_BOX: 2.0,
        BEND_TO_PLACE: 3.0
    }

    # ricette
    RECIPE_NONE   = 0.0
    RECIPE_WEIGHT = 1.0
    RECIPE_BOX    = 2.0
    RECIPE_BOTH   = 3.0

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

        # Telemetria FSM/assistenza
        self.state_id_pub       = self.create_publisher(Float64, 'admittance/fsm_state_id', 10)
        self.assist_active_pub  = self.create_publisher(Bool,   'admittance/assist_active', 10)
        self.assist_recipe_pub  = self.create_publisher(Float64,'admittance/assist_recipe_id', 10)

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
        self.HARD_K_DEFAULT = 0.0
        self.HARD_K_MIN = 10.0
        self.down_K = 0.0
        self.traj_idle_s = 1.0

        # Ramp K/C (smoothstep) + hysteresis
        self.time_set = 0.4
        self.zeta = self.C_ratio
        self.K_current = float(self.K); self.K_start = float(self.K); self.K_target = float(self.K)
        self.C_start = float(self.C);   self.C_target = float(self._c_from_k(self.K))
        self.k_trj_active = False
        self.k_t0 = self.get_clock().now(); self.k_t1 = self.get_clock().now()
        self.HARD_HOLD_S = 0.6
        self.hard_until = self.get_clock().now()
        self.require_still_to_soft = True
        self.min_soft_still_vel = 0.10
        self.mode = "soft"

        # ---------- Rampa τ_ass ----------
        self.tau_time_set    = 0.8
        self.tau_ass_current = 0.0
        self.tau_ass_start   = 0.0
        self.tau_ass_target  = 0.0
        self.tau_trj_active  = False
        self.tau_t0          = self.get_clock().now()
        self.tau_t1          = self.get_clock().now()
        self.tau_resched_eps = 0.3

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
        self.declare_parameter('m_w', (1-0.142-0.1*2-0.06*2) * 70.0)          # [kg] 25
        self.declare_parameter('l_w', (0.72-0.53) * 1.80)          # [m] 0.45
        self.declare_parameter('m_b', 4.0)           # [kg]
        self.declare_parameter('l_int', 0.55)        # [m]
        self.declare_parameter('l_b', 0.20)          # [m]
        self.declare_parameter('assist_max_nm', 200.0)
        self.declare_parameter('coeff_assist', 0.2) # it was 0.5
        self.declare_parameter('theta_r_deadzone', 0.01)
        self.declare_parameter('perception_on', False)    # abilita τ_b
        self.declare_parameter('offset', 0.0)            # offset angolare per modello gravità

        self.declare_parameter('tau_slew_rate', 80.0)  # [Nm/s] velocità massima variazione tau_ass
        self.tau_slew_rate = float(self.get_parameter('tau_slew_rate').value)

        # --- Slew-rate asimmetrico + ease-in sulla SALITA ---
        self.declare_parameter('tau_slew_rate_up',   60.0)   # [Nm/s] salita max
        self.declare_parameter('tau_slew_rate_down', 90.0)   # [Nm/s] discesa max
        self.declare_parameter('tau_rise_ease_s',    0.25)   # [s] durata ease-in in salita
        self.declare_parameter('tau_rise_min_fac',   0.25)   # fattore iniziale (0..1) del rate in salita

        self.tau_slew_rate_up   = float(self.get_parameter('tau_slew_rate_up').value)
        self.tau_slew_rate_down = float(self.get_parameter('tau_slew_rate_down').value)
        self.tau_rise_ease_s    = float(self.get_parameter('tau_rise_ease_s').value)
        self.tau_rise_min_fac   = float(self.get_parameter('tau_rise_min_fac').value)

        # timestamp dell'ultima accensione assist (per ease-in)
        self._assist_on_time = None


        # ---------- Soglie posture + dead-zones ----------
        self.declare_parameter('theta_stand', 0.05)      # [rad]
        self.declare_parameter('theta_bend', 0.55)  # [rad]
        self.declare_parameter('assist_margin', 0.01)   # [rad]
        self.declare_parameter('assist_delay_s', 1.0)
        self.assist_delay_s = float(self.get_parameter('assist_delay_s').value)
        self._rise_from_bend_t0 = None


        # ---------- Step / lead reference ----------
        self.declare_parameter('step_mode', True)
        self.declare_parameter('step_delta', 0.07)      # [rad]
        self.declare_parameter('step_K', 40.0)          # K piccola durante il lead
        self.declare_parameter('step_min_speed', 0.2)  # [rad/s] (solo per lead reference)

        # ---------- Scala τ_b per 2->3 (se vuoi) ----------
        self.declare_parameter('box_scale_23', 0.2) # it was 1.0

        # leggi parametri
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
        self.offset = float(self.get_parameter('offset').value)

        self.theta_stand = float(self.get_parameter('theta_stand').value)
        self.theta_bend  = float(self.get_parameter('theta_bend').value)
        self.assist_margin = float(self.get_parameter('assist_margin').value)

        self.step_mode      = bool(self.get_parameter('step_mode').value)
        self.step_delta     = float(self.get_parameter('step_delta').value)
        self.step_K         = float(self.get_parameter('step_K').value)
        self.step_min_speed = float(self.get_parameter('step_min_speed').value)

        self.box_scale_23   = float(self.get_parameter('box_scale_23').value)

        # ---------- FSM + assist ----------
        self.state = self.STAND_NO_BOX
        self.assist_active = False
        self.assist_recipe_id = self.RECIPE_NONE
        self.assist_until = None

        self.curr_weight_scale = 1.0
        self.curr_box_scale    = 1.0

        # stato box_gate locale
        self.box_gate = False
        self._publish_fsm_and_assist()

        # ---------- Timers ----------
        self.create_timer(self.dt,  self._update_k_profile)
        self.create_timer(self.dt,  self._update_tau_ass_profile)
        self.create_timer(0.1,      self._maybe_revert_to_soft)
        self.create_timer(0.1,      self._publish_params)

        self.get_logger().info("Admittance controller (positional assist) ready.")

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
            s = u*u*(3 - 2*u)
            self.K_current = self.K_start + (self.K_target - self.K_start)*s
            self.C         = self.C_start + (self.C_target - self.C_start)*s
        self.K = self.K_current

    # ===== Rampa τ_ass =====
    def _schedule_tau_ass(self, target: float, duration: float):
        # target = max(-self.assist_max_nm, min(self.assist_max_nm, float(target))) # con clamp 
        target = float(target) # senza clamp
        now = self.get_clock().now()
        self.tau_ass_start  = float(self.tau_ass_current)
        self.tau_ass_target = target
        self.tau_t0 = now
        self.tau_t1 = now + rclpy.duration.Duration(seconds=max(0.05, duration))
        self.tau_trj_active = True

    # def _update_tau_ass_profile(self):
    #     if not self.tau_trj_active:
    #         return
    #     now = self.get_clock().now()
    #     if now >= self.tau_t1:
    #         self.tau_ass_current = self.tau_ass_target
    #         self.tau_trj_active = False
    #     else:
    #         num = (now - self.tau_t0).nanoseconds
    #         den = (self.tau_t1 - self.tau_t0).nanoseconds
    #         u = max(0.0, min(1.0, num/den)) if den > 0 else 1.0
    #         s = u*u*(3 - 2*u)
    #         self.tau_ass_current = self.tau_ass_start + (self.tau_ass_target - self.tau_ass_start)*s

    # def _update_tau_ass_profile(self):
    #     # Slew-rate limiter simmetrico (nessun reschedule, nessun smoothstep)
    #     e = self.tau_ass_target - self.tau_ass_current
    #     if e == 0.0:
    #         return
    #     max_step = self.tau_slew_rate * self.dt
    #     if e > 0.0:
    #         self.tau_ass_current += min(e, max_step)
    #     else:
    #         self.tau_ass_current += max(e, -max_step)

    def _update_tau_ass_profile(self):
        # Errore rispetto al target aggiornato ogni ciclo
        e = self.tau_ass_target - self.tau_ass_current
        if e == 0.0:
            return

        # Scegli rate base: salita vs discesa
        if e > 0.0:
            base_rate = self.tau_slew_rate_up
            # Ease-in solo in SALITA e solo quando l'assistenza è ON
            ease_fac = 1.0
            if self.assist_active and self._assist_on_time is not None and self.tau_rise_ease_s > 1e-6:
                now = self.get_clock().now()
                t_elapsed = (now - self._assist_on_time).nanoseconds * 1e-9
                u = max(0.0, min(1.0, t_elapsed / self.tau_rise_ease_s))
                # smoothstep per arrivare da fac_min -> 1.0
                s = u*u*(3.0 - 2.0*u)
                fac_min = max(0.0, min(1.0, self.tau_rise_min_fac))
                ease_fac = fac_min + (1.0 - fac_min) * s
            eff_rate = base_rate * ease_fac
        else:
            # discesa: nessun ease-in (o aggiungilo anche qui se vuoi)
            eff_rate = self.tau_slew_rate_down

        max_step = eff_rate * self.dt
        if e > 0.0:
            self.tau_ass_current += min(e, max_step)
        else:
            self.tau_ass_current += max(e, -max_step)



    # ===== Supporto: publish stato/assist =====
    def _publish_fsm_and_assist(self):
        self.state_id_pub.publish(Float64(data=self.STATE_ID[self.state]))
        self.assist_active_pub.publish(Bool(data=self.assist_active))
        self.assist_recipe_pub.publish(Float64(data=self.assist_recipe_id))

    # ===== FSM posizionale (cambia stato quando superi le soglie assolute) =====
    def _maybe_transition_state(self, theta_c: float):
        prev = self.state
        if self.state == self.STAND_NO_BOX:
            if theta_c >= self.theta_bend:
                self.state = self.BEND_TO_PICK
        elif self.state == self.BEND_TO_PICK:
            if theta_c <= self.theta_stand:
                self.state = self.STAND_WITH_BOX
        elif self.state == self.STAND_WITH_BOX:
            if theta_c >= self.theta_bend:
                self.state = self.BEND_TO_PLACE
        elif self.state == self.BEND_TO_PLACE:
            if theta_c <= self.theta_stand:
                self.state = self.STAND_NO_BOX
        if self.state != prev:
            self.get_logger().info(f"[FSM] {prev} → {self.state}")
            self._rise_from_bend_t0 = None
            self._publish_fsm_and_assist()
        return prev  # <— restituisco lo stato precedente

    # ===== Assistenza: SOLO per stato/posizione (no velocità) =====
    def _update_assist_by_state(self, theta_c: float, prev_state: str):
        if self.assist_active:
            return  # lo spegnimento è gestito da _maybe_stop_assist()

        # 0->1: nessuna assistenza
        if self.state == self.STAND_NO_BOX:
            return

        # 1->2: attiva se theta_c < theta_bend (sto risalendo dal bend) OPPURE se box_gate è ON
        if self.state == self.BEND_TO_PICK:
            now = self.get_clock().now()

            # Condizione 1: box_gate → ON immediato
            if self.box_gate:
                self._activate_assist(self.RECIPE_BOTH, until='stand')
                self._rise_from_bend_t0 = None
                return

            # Condizione 2: già sotto la soglia di bend → ON immediato
            if theta_c < self.theta_bend:
                self._activate_assist(self.RECIPE_BOTH, until='stand')
                self._rise_from_bend_t0 = None
                return

            # Condizione 3: sopra la soglia da almeno assist_delay_s → ON
            # (cioè: resta in bend_to_pick, ma non è ancora sceso sotto la soglia;
            # dopo assist_delay_s di "attesa" sopra la soglia, attiva assistenza)
            if self._rise_from_bend_t0 is None:
                # primo istante sopra la soglia in stato 1
                self._rise_from_bend_t0 = now
            else:
                dt = (now - self._rise_from_bend_t0).nanoseconds * 1e-9
                if dt >= self.assist_delay_s:
                    self._activate_assist(self.RECIPE_BOTH, until='stand')
                    self._rise_from_bend_t0 = None
            return

        # 2->3: attiva ESATTAMENTE alla transizione (prev=2, now=3)
        if self.state == self.STAND_WITH_BOX:
            if (theta_c > self.theta_stand):
                self._activate_assist(self.RECIPE_BOX, until='bend')
            return


        # 3->0: attiva mentre SEI in 3 (fino a θ_stand)
        # if self.state == self.BEND_TO_PLACE:
        #     # se non è attiva, accendila
        #     if (theta_c < self.theta_bend): 
        #         self._activate_assist(self.RECIPE_WEIGHT, until='stand')
        #     return
        if self.state == self.BEND_TO_PLACE:
            now = self.get_clock().now()

            # Condizione 1: box_gate → ON immediato (non necessario per la ricetta peso,
            # ma lo usiamo come ulteriore "OR" come richiesto)
            #if not self.box_gate:
            if self.box_gate:
                self._activate_assist(self.RECIPE_WEIGHT, until='stand')
                self._rise_from_bend_t0 = None
                return

            # Condizione 2: già sotto la soglia di bend → ON immediato
            # (sei ancora piegato: attiva supporto al peso finché risali a stand)
            if theta_c < self.theta_bend:
                self._activate_assist(self.RECIPE_WEIGHT, until='stand')
                self._rise_from_bend_t0 = None
                return

            # Condizione 3: sopra la soglia di bend da almeno assist_delay_s → ON
            # (sei tra theta_bend e theta_stand da un po’ ma ancora nello stato 3)
            if self._rise_from_bend_t0 is None:
                # primo istante "sopra" in stato 3
                self._rise_from_bend_t0 = now
            else:
                dt = (now - self._rise_from_bend_t0).nanoseconds * 1e-9
                if dt >= self.assist_delay_s:
                    self._activate_assist(self.RECIPE_WEIGHT, until='stand')
                    self._rise_from_bend_t0 = None
            return


    def _activate_assist(self, recipe_id: float, until: str, weight_scale: float = 1.0, box_scale: float = 1.0):
        self._assist_on_time = self.get_clock().now()
        self.assist_active = True
        self.assist_recipe_id = recipe_id  # 0 none, 1 weight, 2 box, 3 both
        self.assist_until = until          # 'stand' o 'bend'
        self.curr_weight_scale = float(weight_scale)
        self.curr_box_scale    = float(box_scale)
        self.get_logger().info(f"[ASSIST] ON recipe={recipe_id} until={until} (w_scale={self.curr_weight_scale}, b_scale={self.curr_box_scale})")
        self._publish_fsm_and_assist()

    def _deactivate_assist(self):
        self.assist_active = False
        self.assist_recipe_id = self.RECIPE_NONE
        self.assist_until = None
        self.curr_weight_scale = 1.0
        self.curr_box_scale    = 1.0
        self.get_logger().info("[ASSIST] OFF (reached target)")

        # self._schedule_tau_ass(0.0, self.tau_time_set)
        self.tau_ass_target = 0.0

        self._publish_fsm_and_assist()

    def _maybe_stop_assist(self, theta_c: float):
        if not self.assist_active:
            return
        # Spegni quando sei vicino al target E nel nuovo stato
        if self.assist_until == 'stand':
            close = abs(theta_c - self.theta_stand) <= self.assist_margin
            in_new_state = (self.state in (self.STAND_WITH_BOX, self.STAND_NO_BOX))
            if close and in_new_state:
                self._deactivate_assist()
        elif self.assist_until == 'bend':
            close = abs(theta_c - self.theta_bend) <= self.assist_margin
            in_new_state = (self.state in (self.BEND_TO_PICK, self.BEND_TO_PLACE))
            if close and in_new_state:
                self._deactivate_assist()



    # ===== callback su box_gate =====
    def box_gate_callback(self, msg: Bool):
        new_val = bool(msg.data)
        # fronte di salita: False -> True → accendi subito in stato 1
        if (not self.box_gate) and new_val:
            if (self.state == self.BEND_TO_PICK) and (not self.assist_active):
                self._activate_assist(self.RECIPE_BOTH, until='stand')

        # fronte di discesa: True -> False → ON in stato 3 (rilascio pacco)
        if self.box_gate and (not new_val):
            if (self.state == self.BEND_TO_PLACE) and (not self.assist_active):
                self._activate_assist(self.RECIPE_WEIGHT, until='stand')

        self.box_gate = new_val

    # ===== Callbacks vari =====
    def theta_ref_callback(self, msg: Float64):
        now = self.get_clock().now()
        was_idle = (now - self.last_theta_ref_time).nanoseconds * 1e-9 > self.traj_idle_s
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

    def _set_hard(self, K=None, duration=None):
        duration = self.time_set if duration is None else duration
        if self.mode == "hard" and (K is None or abs(self.K_target - K) < 1e-6):
            return
        self.mode = "hard"
        K = self.HARD_K_DEFAULT if K is None else K
        self._schedule_k(K, duration)
        self.hard_until = self.get_clock().now() + rclpy.duration.Duration(seconds=self.HARD_HOLD_S)

    def _set_soft(self, duration=None):
        duration = self.time_set if duration is None else duration
        if self.mode == "soft" and abs(self.K_target - self.SOFT_K) < 1e-6 and not self.k_trj_active:
            return
        self.mode = "soft"
        self._schedule_k(self.SOFT_K, duration)

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
        for i in range(2):
            zeroed_effort = msg.effort[i] - self.measured_eff_zero[i]
            self.filtered_p[i] = self.alpha*msg.position[i] + (1 - self.alpha)*self.filtered_p[i]
            self.filtered_v[i] = self.alpha*msg.velocity[i] + (1 - self.alpha)*self.filtered_v[i]
            self.filtered_e[i] = self.alpha_eff*zeroed_effort + (1 - self.alpha_eff)*self.filtered_e[i]

    def _theta_central(self) -> float:
        # centrale ≈ 0.5*(right - left)
        return 0.5 * (self.filtered_p[0] - self.filtered_p[1])

    def _compute_tau_components(self, theta_w: float):
        # offset per eventuale calibrazione zero postura
        tau_w = self.m_w * self.g * self.l_w * math.sin(theta_w + self.offset)
        tau_box = self.m_b * self.g * (self.l_int * math.sin(theta_w + self.offset) + self.l_b)
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

        # filtri
        self.state_filter(msg)

        # centrale e velocità (vel solo per step mode)
        theta_c     = self._theta_central()
        theta_c_dot = 0.5 * (self.filtered_v[0] - self.filtered_v[1])

        # step mode classica (identica)
        if self.step_mode and abs(theta_c_dot) > self.step_min_speed:
            dir_sign = 1 if theta_c_dot > 0 else -1
            theta_ref_c = theta_c + dir_sign * self.step_delta
            self.x_ref = [theta_ref_c, -theta_ref_c]
            self._set_hard(K=self.step_K, duration=self.time_set)

        # FSM posizionale (annota prev_state per i trigger “esatti”)
        prev_state = self._maybe_transition_state(theta_c)

        # Attiva assistenza in base a stato/posizione (usa anche prev_state)
        self._update_assist_by_state(theta_c, prev_state)

        # Componenti gravitazionali (centrali)
        tau_w, tau_b_raw = self._compute_tau_components(theta_c)

        # # τ_b applicata solo se perception_on e box_gate sono veri
        # tau_b = (tau_b_raw if (self.perception_on and self.box_gate) else 0.0)
        # DOPO: calcolo sempre e pubblico SEMPRE il valore fisico calcolato
        tau_b = tau_b_raw
        self.tau_box_pub.publish(Float64(data=float(tau_b)))

        # Se assist attiva, seleziona ricetta; altrimenti 0
        tau_target = 0.0
        if self.assist_active:

            # versione per la ricetta: entra solo se perception_on è True
            tau_b_recipe = tau_b if self.perception_on else 0.0

            if self.assist_recipe_id in (self.RECIPE_WEIGHT, self.RECIPE_BOTH):
                tau_target += self.curr_weight_scale * tau_w
            if self.assist_recipe_id in (self.RECIPE_BOX, self.RECIPE_BOTH):
                # tau_target += self.curr_box_scale * tau_b
                tau_target += self.curr_box_scale * tau_b

            # Deadzone vicino a θ_r≈0 (anti-jitter)
            theta_r_meas = self.filtered_p[0]
            if abs(theta_r_meas) <= self.theta_r_deadzone:
                tau_target = 0.0

            # Spegnimento assist “on target & in new state”
            self._maybe_stop_assist(theta_c)
            if not self.assist_active:
                tau_target = 0.0  # nel loop in cui spengo

        # Gain + clamp
        tau_target *= self.coeff_assist
        # tau_target = max(-self.assist_max_nm, min(self.assist_max_nm, tau_target)) # per il clamp, se no commenta questa riga

        # Rampa τ_ass se cambia abbastanza
        # if (abs(tau_target - self.tau_ass_target) > self.tau_resched_eps) or \
        #    (not self.tau_trj_active and abs(tau_target - self.tau_ass_current) > self.tau_resched_eps):
        #     self._schedule_tau_ass(tau_target, self.tau_time_set)
        self.tau_ass_target = float(tau_target)

        # τ_ass applicata (centrale) e per-giunto
        tau_ass_applied = self.tau_ass_current
        tau_ass_L = -tau_ass_applied
        tau_ass_R =  tau_ass_applied

        # Dinamica virtuale
        for i, tau_ass_i in enumerate([tau_ass_L, tau_ass_R]):
            self.a[i] = (tau_ass_i + self.filtered_e[i] - self.C*self.v[i] - self.K*(self.x[i] - self.x_ref[i])) / self.M
            self.v[i] += self.a[i] * self.dt
            self.x[i] += self.v[i] * self.dt

        # Vincolo L=-R
        theta_ref_l = -self.x[1]
        theta_ref_r =  self.x[1]

        # Pubblicazioni
        self.position_pub.publish(Float32MultiArray(data=[theta_ref_l, theta_ref_r]))
        self.est_torque_pub.publish(Float32MultiArray(data=self.filtered_e))
        self.tau_meas_pub.publish(Float32MultiArray(data=self.filtered_e))
        self.tau_ass_pub.publish(Float32MultiArray(data=[tau_ass_L, tau_ass_R]))
        self.tau_w_pub.publish(Float64(data=float(tau_w)))
        self.tau_box_pub.publish(Float64(data=float(tau_b)))
        self.tau_ass_total_pub.publish(Float64(data=float(tau_ass_applied)))

    # ===== Policy ritorno a soft =====
    def _maybe_revert_to_soft(self):
        now = self.get_clock().now()
        dt_idle = (now - self.last_theta_ref_time).nanoseconds * 1e-9
        if dt_idle >= self.traj_idle_s:
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
