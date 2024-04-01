import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

# Define the bandit reward distributions with mean 0 and variance 1
def generate_bandit_rewards(num_arms, num_bandit_problems):
    bandit_reward_distributions = [
        [np.random.normal(0, 1) for _ in range(num_bandit_problems)]
        for _ in range(num_arms)
    ]
    return bandit_reward_distributions

num_arms = 10
num_bandit_problems = 2000
bandit_reward_distributions = generate_bandit_rewards(num_arms, num_bandit_problems)

def pull_arm(arm, bandit_rewards):
    reward = bandit_rewards[arm]
    return reward

def epsilon_greedy(epsilon, num_episodes, bandit_rewards, decay_epsilon = False):
    q_values = np.random.normal(size = num_arms)
    action_counts = np.zeros(num_arms)
    rewards = []

    for _ in range(num_episodes):
        if np.random.rand() < epsilon:
            arm = np.random.choice(num_arms)
        else:
            arm = np.argmax(q_values)
        
        reward = pull_arm(arm, bandit_rewards[arm])
        action_counts[arm] += 1

        q_values[arm] = q_values[arm] + (1/action_counts[arm])*(reward - q_values[arm])

        rewards.append(reward)

        if decay_epsilon:
            epsilon = epsilon*decay_epsilon

    return rewards

if __name__ == "__main__":
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.3] #np.arange(0.1, 0.6, 0.1)
    num_episodes = 1000

    for epsilon in epsilons:
        rwrds = np.zeros(num_episodes)
        for _ in range(2000):
            rewards = epsilon_greedy(epsilon=epsilon, num_episodes=num_episodes, decay_epsilon=None, bandit_rewards = bandit_reward_distributions)
            rwrds += np.array(rewards)
            
        rwrds = rwrds/2000
        plt.plot(rwrds, label = f"epsilon = {epsilon:.2f}", linewidth = 3)

    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title(f"Epsilon-Greedy Bandit")
    plt.legend()
    plt.show()

    
