from src.estimator.ExtremlyRandomizeTreeEstimator import ExtremlyRandomizeTreeEstimator
from src.estimator.LinearRegressionEstimator import LinearRegressionEstimator
from src.estimator.NeuralNetworkEstimator import NeuralNetworkEstimator
from src.hillproblem.Agent import Agent
from src.hillproblem.HillDomain import HillDomain
from src.policy.Policy import Policy


my_domain = HillDomain()
my_policy = Policy()
est = NeuralNetworkEstimator
my_agent = Agent(my_policy, est)

my_agent.policy = Policy("right")
my_agent.play()
my_agent.policy = Policy("left")
my_agent.play()
my_agent.policy = Policy()
for _ in range(20):
    my_agent.play()

my_agent.action_state_compute_estimator(25)
print('estmitaor done')
my_agent.policy_set_optimal()
print('policy done')
my_agent.play(display=True);
my_agent.action_state_display(25)
print('Program Ended')
