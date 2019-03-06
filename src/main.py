from src.hillproblem.HillDomain import HillDomain
from src.policy.Policy import Policy
from src.hillproblem.Agent import Agent


my_domain = HillDomain()
my_policy = Policy()
my_agent = Agent(my_policy)


my_agent.play(display=True, animation=True)
my_agent.expected_return_compute(10000, True)
