"""
PID Controller — pure control theory, no LLM dependencies.

Used by PIDScheduler to regulate KV cache admission budget.
"""


class PIDController:
    """Discrete PID controller with anti-windup clamping.

    Computes: output = Kp * error + Ki * integral(error) + Kd * d(error)/dt
    where error = setpoint - process_variable.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.01,
        kd: float = 0.1,
        setpoint: float = 0.0,
        output_min: float = 0.0,
        output_max: float = float('inf'),
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_min = output_min
        self.output_max = output_max

        # Internal state
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, pv: float) -> float:
        """Compute PID output given current process variable.

        Args:
            pv: current process variable (e.g., total KV tokens used)

        Returns:
            Control output (e.g., admission budget in tokens)
        """
        error = self.setpoint - pv

        # Integral with anti-windup clamping
        self.integral += error
        self.integral = max(-self.output_max, min(self.output_max, self.integral))

        # Derivative
        derivative = error - self.prev_error
        self.prev_error = error

        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(self.output_min, min(self.output_max, output))

    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
