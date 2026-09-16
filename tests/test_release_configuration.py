import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ReleaseConfigurationTest(unittest.TestCase):
    def test_compose_pins_dual_go2_v128_behavior(self):
        compose = (ROOT / "compose.run1-v128.yml").read_text(encoding="utf-8")
        self.assertIn("RACER_BOTS: \"1,2\"", compose)
        self.assertIn("platform_mode:=ground_omni", compose)
        self.assertIn("RACER_CONSISTENCY_SIGN: \"1.0\"", compose)
        self.assertIn("RACER_FIRST_GRID_BONUS: \"6.0\"", compose)
        self.assertIn("bootstrap_directions_deg:='[90.0]'", compose)
        self.assertNotIn(":latest", compose)

    def test_go2_has_the_validated_retro_reflector(self):
        xacro = (
            ROOT / "src/legged/legged_sim/xacro/go2_mid360.xacro"
        ).read_text(encoding="utf-8")
        self.assertIn('<link name="retro_beacon_link">', xacro)
        self.assertIn("<laser_retro>5000</laser_retro>", xacro)

    def test_recorder_uses_simulation_time_and_long_tracking_timeout(self):
        launcher = (
            ROOT / "scripts/run_legged_dual_exploration.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--duration-basis sim", launcher)
        self.assertIn("--wait-for-tracking", launcher)
        self.assertIn("--startup-timeout 1800", launcher)
        self.assertIn("auto_trot true", launcher)
        self.assertIn("grep -qi 'successful'", launcher)
        self.assertIn("trap cleanup_on_exit EXIT", launcher)

    def test_v128_hgrid_has_per_vehicle_first_grid_bonus(self):
        hgrid = (
            ROOT
            / "src/RACER/swarm_exploration/active_perception/src/hgrid.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("first_grid_bonus_", hgrid)
        self.assertIn("drone_index", hgrid)


if __name__ == "__main__":
    unittest.main()
