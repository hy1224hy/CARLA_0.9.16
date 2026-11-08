"""Avoid repeated map loads while retaining idle-world and scoped ownership rules."""
import unittest
from unittest.mock import MagicMock,patch


class WorldReuseTests(unittest.TestCase):
    def fixture(self,name='/Game/Carla/Maps/Town05',busy=False):
        client=MagicMock();world=client.get_world.return_value
        world.get_map.return_value.name=name
        world.get_actors.return_value.filter.side_effect=lambda pattern: [object()] if busy and pattern=='vehicle.*' else []
        return client,world

    def test_same_idle_town_is_kept_without_tick_or_load(self):
        import carla
        from utils.evaluate_mut_mappo_critical import connect_reusing_idle_world
        client,world=self.fixture()
        with patch.object(carla,'Client',return_value=client):
            self.assertEqual(connect_reusing_idle_world('127.0.0.1',2000,'Town05'),(client,world))
        client.load_world.assert_not_called();world.tick.assert_not_called()
        world.apply_settings.assert_not_called()

    def test_different_town_loads_the_requested_map(self):
        import carla
        from utils.evaluate_mut_mappo_critical import connect_reusing_idle_world
        client,_=self.fixture('/Game/Carla/Maps/Town10HD_Opt')
        with patch.object(carla,'Client',return_value=client):
            _,world=connect_reusing_idle_world('127.0.0.1',2000,'Town05')
        client.load_world.assert_called_once_with('Town05')
        self.assertIs(world,client.load_world.return_value)

    def test_busy_world_is_rejected_before_any_map_change(self):
        import carla
        from utils.evaluate_mut_mappo_critical import connect_reusing_idle_world
        client,_=self.fixture(busy=True)
        with patch.object(carla,'Client',return_value=client):
            with self.assertRaisesRegex(RuntimeError,'idle world'):
                connect_reusing_idle_world('127.0.0.1',2000,'Town05')
        client.load_world.assert_not_called()

    def test_operational_override_is_scoped_even_on_failure(self):
        from utils.evaluate_mut_mappo_critical import geometry_capture,connect_reusing_idle_world
        from environment import carla_multi_agent_env as core
        original=core._connect_and_load_world
        with self.assertRaisesRegex(RuntimeError,'test interruption'):
            with geometry_capture(reuse_idle_world=True):
                self.assertIs(core._connect_and_load_world,connect_reusing_idle_world)
                raise RuntimeError('test interruption')
        self.assertIs(core._connect_and_load_world,original)


if __name__=='__main__':unittest.main()
