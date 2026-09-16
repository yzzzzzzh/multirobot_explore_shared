#!/usr/bin/env python3
"""Bridge geometry_msgs/Twist (cmd_vel) to the unitree_guide_controller input.

The guide controller is driven like a joystick: control_input_msgs/Inputs with a
state command (1 passive, 2 fixed down/stand, 4 trotting, 0 none) and four stick
axes in [-1, 1].  In the trotting state the axes map to body velocities
(StateTrotting.cpp): vx = invNormalize(ly), vy = -invNormalize(lx),
yaw_rate = -invNormalize(rx), each over a symmetric limit.

The controller publishes its FSM state on `fsm_state`; this node drives the
robot through passive -> fixed down -> fixed stand -> trotting with short
command pulses (the FSM reacts to the command level, so a held command would
toggle states) and then streams velocities.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from control_input_msgs.msg import Inputs


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# desired next command for each FSM state on the way to trotting
NEXT_COMMAND = {"trotting": {"passive": 2, "fixed down": 2, "fixed stand": 4},
                "rl": {"passive": 2, "fixed down": 2, "fixed stand": 3}}


class TwistToControlInput(Node):
    def __init__(self):
        super().__init__("twist_to_control_input")
        self.vx_limit = float(self.declare_parameter("vx_limit", 0.4).value)
        self.vy_limit = float(self.declare_parameter("vy_limit", 0.3).value)
        self.wz_limit = float(self.declare_parameter("wz_limit", 0.5).value)
        self.auto_trot = bool(self.declare_parameter("auto_trot", True).value)
        # "trotting" (unitree_guide: fixed stand -> 4) or "rl" (rl controller: fixed stand -> 3)
        self.walk_state = str(self.declare_parameter("walk_state", "trotting").value)
        # RL policies take velocities in m/s and rad/s directly; the guide trot takes
        # normalised [-1, 1] sticks scaled by its limits.
        self.raw_velocity_commands = bool(self.declare_parameter("raw_velocity_commands", False).value)
        # The RL policy stops stepping below roughly 0.25 m/s; a yaw command
        # without stepping winds the stance legs up to the hip limits and the
        # robot trips.  When > 0, any turn or slow-advance request is given at
        # least this much forward speed so the gait stays active (the robot
        # arcs instead of turning in place).  0 disables the shaping.
        self.min_step_speed = float(self.declare_parameter("min_step_speed", 0.0).value)
        # The RL policy standing still after a motion segment sometimes lets a
        # foot creep outward until the front legs splay and the robot flips
        # (never seen from the position-locked fixed stand).  Park in fixed
        # stand after idle_park_seconds without a motion command and re-enter
        # the gait when the next command arrives (~1 s latency).
        self.park_when_idle = bool(self.declare_parameter("park_when_idle", False).value)
        self.idle_park_seconds = float(self.declare_parameter("idle_park_seconds", 3.0).value)
        # Slew-rate limit on the gait command.  An abrupt stop from 0.5 m/s
        # (command zeroed by a failsafe/replan) made the stance collapse within
        # 2 s; ramping the velocity down lets the policy settle.  0 disables.
        self.max_linear_accel = float(self.declare_parameter("max_linear_accel", 0.0).value)
        self.max_angular_accel = float(self.declare_parameter("max_angular_accel", 0.0).value)
        self.slew_cmd = [0.0, 0.0, 0.0]   # vx, vy, wz actually sent last tick
        self.cmd_timeout = float(self.declare_parameter("cmd_timeout", 0.5).value)
        self.rate_hz = float(self.declare_parameter("rate_hz", 50.0).value)
        self.pulse_seconds = float(self.declare_parameter("pulse_seconds", 0.3).value)
        self.settle_seconds = float(self.declare_parameter("settle_seconds", 2.5).value)
        self.pub = self.create_publisher(Inputs, "control_input", 10)
        self.sub = self.create_subscription(Twist, "cmd_vel", self.on_twist, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.state_sub = self.create_subscription(String, "fsm_state", self.on_state, latched)
        self.twist = Twist()
        self.twist_stamp = None
        self.fsm_state = None
        self.state_since = None
        self.pulse_until = None
        self.pulse_cmd = 0
        self.last_pulse = None
        self.parked = False
        self.idle_since = None
        self.timer = self.create_timer(1.0 / self.rate_hz, self.tick)

    def on_twist(self, msg: Twist):
        self.twist = msg
        self.twist_stamp = self.get_clock().now()

    def on_state(self, msg: String):
        if msg.data != self.fsm_state:
            self.get_logger().info("controller FSM state: %s" % msg.data)
            self.fsm_state = msg.data
            self.state_since = self.get_clock().now()

    def tick(self):
        now = self.get_clock().now()
        auto_trot = bool(self.get_parameter("auto_trot").value)
        if auto_trot != self.auto_trot:
            self.get_logger().info("auto_trot -> %s" % auto_trot)
            self.auto_trot = auto_trot
        msg = Inputs()
        fresh = self.twist_stamp is not None and \
            (now - self.twist_stamp).nanoseconds * 1e-9 <= self.cmd_timeout
        moving = fresh and (abs(self.twist.linear.x) + abs(self.twist.linear.y) +
                            abs(self.twist.angular.z)) > 0.02
        if moving:
            self.idle_since = None
            if self.parked:
                self.get_logger().info("motion command -> leaving parked stand")
                self.parked = False
        elif self.idle_since is None:
            self.idle_since = now
        next_command = NEXT_COMMAND[self.walk_state]
        target_reached = self.fsm_state == self.walk_state or \
            (not self.auto_trot and self.fsm_state == "fixed stand") or \
            (self.parked and self.fsm_state == "fixed stand")
        if self.pulse_until is not None and now < self.pulse_until:
            msg.command = self.pulse_cmd
        elif self.park_when_idle and self.auto_trot and not self.parked and \
                self.fsm_state == self.walk_state and self.idle_since is not None and \
                (now - self.idle_since).nanoseconds * 1e-9 >= self.idle_park_seconds:
            # rl -> (2) fixed down -> (2) fixed stand; the sequencer below is
            # held at fixed stand by self.parked until a motion command arrives.
            self.parked = True
            self.pulse_cmd = 2
            self.pulse_until = now + rclpy.duration.Duration(seconds=self.pulse_seconds)
            self.last_pulse = now
            msg.command = self.pulse_cmd
            self.get_logger().info("idle %.1fs in %s -> parking (command 2)" %
                                   ((now - self.idle_since).nanoseconds * 1e-9, self.fsm_state))
        elif self.fsm_state in next_command and not target_reached:
            settled = self.state_since is not None and \
                (now - self.state_since).nanoseconds * 1e-9 >= self.settle_seconds
            pulse_ok = self.last_pulse is None or \
                (now - self.last_pulse).nanoseconds * 1e-9 >= self.settle_seconds
            if settled and pulse_ok:
                self.pulse_cmd = next_command[self.fsm_state]
                self.pulse_until = now + rclpy.duration.Duration(seconds=self.pulse_seconds)
                self.last_pulse = now
                msg.command = self.pulse_cmd
                self.get_logger().info("state %s -> pulse command %d" % (self.fsm_state, self.pulse_cmd))
        if self.fsm_state == self.walk_state:
            vx = vy = wz = 0.0
            if fresh:
                vx = float(self.twist.linear.x)
                vy = float(self.twist.linear.y)
                wz = float(self.twist.angular.z)
                if self.min_step_speed > 0.0 and abs(vx) < self.min_step_speed:
                    if abs(wz) > 0.05:
                        vx = self.min_step_speed if vx >= 0.0 else -self.min_step_speed
                    elif abs(vx) > 0.05:
                        vx = self.min_step_speed if vx > 0.0 else -self.min_step_speed
            if self.max_linear_accel > 0.0 or self.max_angular_accel > 0.0:
                dt = 1.0 / self.rate_hz
                lin_step = self.max_linear_accel * dt if self.max_linear_accel > 0.0 else 1e9
                ang_step = self.max_angular_accel * dt if self.max_angular_accel > 0.0 else 1e9
                vx = clamp(vx, self.slew_cmd[0] - lin_step, self.slew_cmd[0] + lin_step)
                vy = clamp(vy, self.slew_cmd[1] - lin_step, self.slew_cmd[1] + lin_step)
                wz = clamp(wz, self.slew_cmd[2] - ang_step, self.slew_cmd[2] + ang_step)
                # Keep the gait stepping while decelerating: a ramp through the
                # no-step band is exactly the regime that trips the policy.
                if self.min_step_speed > 0.0 and 0.05 < abs(vx) < self.min_step_speed and \
                        abs(self.slew_cmd[0]) >= self.min_step_speed:
                    vx = self.min_step_speed if vx > 0.0 else -self.min_step_speed
                    if (not fresh) or (abs(float(self.twist.linear.x)) < 0.05 and abs(wz) < 0.05):
                        vx = 0.0   # commanded (or timed-out) stop: drop straight from the step floor
            self.slew_cmd = [vx, vy, wz]
            if self.raw_velocity_commands:
                msg.ly = clamp(vx, -self.vx_limit, self.vx_limit)
                msg.lx = clamp(-vy, -self.vy_limit, self.vy_limit)
                msg.rx = clamp(-wz, -self.wz_limit, self.wz_limit)
            else:
                msg.ly = clamp(vx / self.vx_limit, -1.0, 1.0)
                msg.lx = clamp(-vy / self.vy_limit, -1.0, 1.0)
                msg.rx = clamp(-wz / self.wz_limit, -1.0, 1.0)
        else:
            self.slew_cmd = [0.0, 0.0, 0.0]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TwistToControlInput()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
