import unittest
from servedistill.train import alert

class MonitoringTests(unittest.TestCase):
    def test_sft_joint_trend_does_not_require_rl_reference_metrics(self):
        baseline={"reward_mean":1.0,"entropy":1.0,"gradient_norm":1.0}
        metrics={"behavior_max_error":0.0,"behavior_tolerance":0.15,"reward_mean":0.2,
                 "entropy":0.1,"gradient_norm":0.2,"truncation_rate":0.0,
                 "json_valid":1.0,"effective_group_fraction":0.0}
        self.assertEqual(alert(metrics,[baseline]),[])
        metrics["gradient_norm"]=6.0
        self.assertIn("reward_entropy_divergence_joint_alarm",alert(metrics,[baseline]))
    def test_probability_gate_remains_fatal(self):
        with self.assertRaisesRegex(RuntimeError,"logprob contract"):
            alert({"behavior_max_error":0.16,"behavior_tolerance":0.15},[])

if __name__ == "__main__":unittest.main()
