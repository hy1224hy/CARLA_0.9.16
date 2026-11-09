import unittest
import numpy as np
import torch
from models.actor_critic import Actor
from training.mutual_distillation import build_distillation_gates, student_teacher_kl, validate_distillation_options


class DistillationControlsTests(unittest.TestCase):
    def test_directions_and_teacher_gradient_are_explicit(self):
        student, teacher = Actor(20, 2), Actor(20, 2)
        with torch.no_grad():
            teacher.log_std.fill_(0.3)
            teacher.mu_head.bias.fill_(0.5)
        obs = torch.randn(5, 20)
        forward = student_teacher_kl(student, teacher, obs)
        reverse = student_teacher_kl(student, teacher, obs, direction="teacher_to_student")
        symmetric = student_teacher_kl(student, teacher, obs, direction="symmetric")
        self.assertFalse(torch.allclose(forward, reverse))
        self.assertTrue(torch.allclose(symmetric, (forward+reverse)/2))
        reverse.backward()
        self.assertTrue(any(p.grad is not None for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_matched_controls_keep_valid_row_mass_and_rng_replay(self):
        evidence = np.array([0, 0, 1, 0, 1, 1, 0, 0], dtype=np.float32)
        rng = np.random.RandomState(42)
        state = rng.get_state()
        first = build_distillation_gates(evidence, "random", rng)
        rng.set_state(state)
        np.testing.assert_array_equal(first, build_distillation_gates(evidence, "random", rng))
        self.assertEqual(float(first.sum()), float(evidence.sum()))
        uniform = build_distillation_gates(evidence, "strength")
        self.assertTrue(np.all(uniform == evidence.mean()))
        self.assertEqual(float(uniform.sum()), float(evidence.sum()))
        student, teacher = Actor(20, 2), Actor(20, 2)
        obs = torch.randn(8, 20)
        self.assertTrue(torch.allclose(student_teacher_kl(student, teacher, obs, uniform),
                                      student_teacher_kl(student, teacher, obs) * float(evidence.mean())))

    def test_zero_gate_keeps_denominator_and_forbids_enabled_only(self):
        student, teacher = Actor(20, 2), Actor(20, 2)
        obs = torch.randn(6, 20)
        weights = torch.tensor([0., 0., 1., 0., 0., 0.])
        selected = student_teacher_kl(student, teacher, obs, weights)
        self.assertTrue(torch.allclose(selected, student_teacher_kl(student, teacher, obs[2:3])/6))
        zero = student_teacher_kl(student, teacher, obs, torch.zeros(6))
        self.assertEqual(zero.item(), 0.0)
        with self.assertRaises(ValueError):
            validate_distillation_options({"reduction": "enabled_only"})


if __name__ == "__main__":
    unittest.main()
