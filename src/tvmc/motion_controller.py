import rospy
import roslaunch
from rospy import Publisher
from .enums import DoF, Command, ControlMode
import rose_tvmc_msg.msg as msg
from threading import Thread
from time import sleep


PREFIX = "/rose_tvmc/control"
PACKAGE = "rose_tvmc"
TVMC_EXECUTABLE = "tvmc"
PWMC_EXECUTABLE = "pwmc"


class MotionController:
    def __init__(self) -> None:
        # init control modes
        self.controlModes = dict.fromkeys(DoF, ControlMode.OPEN_LOOP)

        # initialize node
        rospy.init_node("motion_controller")

        # create publishers
        self._thruster_pub = Publisher(f"{PREFIX}/thrust", msg.Thrust, queue_size=50)
        self._command_pub = Publisher(f"{PREFIX}/command", msg.Command, queue_size=50)
        self._control_mode_pub = Publisher(
            f"{PREFIX}/control_mode", msg.ControlMode, queue_size=50
        )
        self._current_state_pub = Publisher(
            f"{PREFIX}/current_point", msg.CurrentPoint, queue_size=50
        )
        self._pid_constants_pub = Publisher(
            f"{PREFIX}/pid_constants", msg.PidConstants, queue_size=50
        )
        self._pid_limits_pub = Publisher(
            f"{PREFIX}/pid_limits", msg.PidLimits, queue_size=50
        )
        self._target_state_pub = Publisher(
            f"{PREFIX}/target_point", msg.TargetPoint, queue_size=50
        )

        # create spinner thread
        self._spinner = Thread(target=self._spin, daemon=True)
        self._spinner.start()

        # ensure graceful shutdown
        self._tvmc_process = None
        self._pwmc_process = None
        rospy.on_shutdown(self.__del__)

    def _spin(self) -> None:
        rospy.spin()

    def start(self) -> None:
        tvmc_node = roslaunch.core.Node(PACKAGE, TVMC_EXECUTABLE)
        pwmc_node = roslaunch.core.Node(PACKAGE, PWMC_EXECUTABLE)

        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()

        self._pwmc_process = launch.launch(pwmc_node)
        self._tvmc_process = launch.launch(tvmc_node)

        # sleep for a second to ensure nodes are all started up
        sleep(1)

    def set_thrust(self, dof: DoF, thrust: float) -> None:
        # ensure control mode is in open loop mode
        if self.controlModes[dof] == ControlMode.CLOSED_LOOP:
            raise AssertionError(
                f"Cannot set thrust for DoF {dof} in closed loop mode."
            )

        t = msg.Thrust()
        t.DoF = dof.value
        t.Thrust = thrust

        self._thruster_pub.publish(t)

    def set_control_mode(self, dof: DoF, control: ControlMode) -> None:
        mode = msg.ControlMode()
        mode.DoF = dof.value
        mode.Mode = control.value

        self._control_mode_pub.publish(mode)
        self.controlModes[dof] = mode

    def send_command(self, command: Command) -> None:
        command_msg = msg.Command()
        command_msg.Command = command.value

        self._command_pub.publish(command_msg)

    def set_current_point(self, dof: DoF, current_point: float):
        if self.controlModes[dof] == ControlMode.OPEN_LOOP:
            raise AssertionError(
                "Cannot set current state for DoF {dof} in open loop mode."
            )

        point = msg.CurrentPoint()
        point.DoF = dof.value
        point.Current = current_point

        self._current_state_pub.publish(point)

    def set_pid_constants(
        self, dof: DoF, kp: float, ki: float, kd: float, acceptable_error: float
    ) -> None:
        pidMsg = msg.PidConstants()
        pidMsg.DoF = dof.value
        pidMsg.Kp = kp
        pidMsg.Ki = ki
        pidMsg.Kd = kd
        pidMsg.AcceptableError = acceptable_error

        self._pid_constants_pub.publish(pidMsg)

    def set_pid_limits(
        self,
        dof: DoF,
        integral_min: float,
        integral_max: float,
        output_min: float,
        output_max: float,
    ) -> None:
        limits = msg.PidLimits()
        limits.DoF = dof.value
        limits.IntegralMax = integral_max
        limits.IntegralMin = integral_min
        limits.OutputMax = output_max
        limits.OutputMin = output_min

        self._pid_limits_pub.publish(limits)

    def set_target_point(self, dof: DoF, target: float) -> None:
        if self.controlModes[dof] == ControlMode.OPEN_LOOP:
            raise AssertionError(
                "Cannot set target point for DoF {dof} in open loop mode."
            )

        point = msg.TargetPoint()
        point.DoF = dof.value
        point.Target = target

        self._target_state_pub.publish(point)

    def __del__(self):
        # if TVMC/PWMC was started by this node, kill it as well
        if self._tvmc_process:
            shutdown = msg.Command()
            shutdown.Command = Command.SHUT_DOWN.value
            self._command_pub.publish(shutdown)

        if self._pwmc_process and self._pwmc_process.is_alive():
            self._pwmc_process.stop()
