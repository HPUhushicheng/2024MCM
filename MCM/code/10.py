import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis, :])
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:  # 检查内存中是否有足够的样本
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :])[0])
            target_f = self.model.predict(state[np.newaxis, :])
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



class Environment:
    def __init__(self):
        self.state_size = 4
        self.action_size = 2
        self.agents = [DQNAgent(self.state_size, self.action_size) for _ in range(3)]
        self.episode_rewards = []

    def step(self, actions):
        rewards = [0] * len(self.agents)
        for i, agent in enumerate(self.agents):
            state = np.random.rand(self.state_size)
            next_state = np.random.rand(self.state_size)
            reward = np.random.rand()
            agent.remember(state, actions[i], reward, next_state, False)
            rewards[i] = reward
        return rewards

    def run(self, episodes=70):
        episode_rewards = []
        for e in range(episodes):
            actions = [agent.act(np.random.rand(self.state_size)) for agent in self.agents]
            rewards = self.step(actions)
            episode_rewards.extend(rewards)
            self.episode_rewards.append(sum(rewards))
            for i, agent in enumerate(self.agents):
                agent.replay(32)
            if e % 100 == 0:
                print("Episode: {}, Rewards: {}".format(e, rewards))

        # 绘制累积奖励曲线
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(episodes), self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward over Episodes')

        # 绘制奖励分布直方图
        plt.subplot(1, 2, 2)
        plt.hist(episode_rewards, bins=20)
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.tight_layout()
        plt.show()

env = Environment()
env.run()
