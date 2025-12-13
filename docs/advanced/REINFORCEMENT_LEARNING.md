# Reinforcement Learning Integration

This document describes the reinforcement learning (RL) integration in the 4D Neural Cognition system.

## Overview

The `ReinforcementLearningIntegrator` class in `motor_output.py` provides a comprehensive framework for integrating reinforcement learning algorithms with neural networks. It supports multiple RL algorithms and can be used for motor control, decision-making, and adaptive behavior.

## Supported Algorithms

### 1. Temporal Difference (TD) Learning

Basic value-based learning that estimates state values through experience.

```python
from src.motor_output import ReinforcementLearningIntegrator

# Initialize with TD learning
rl = ReinforcementLearningIntegrator(
    learning_rate=0.01,
    discount_factor=0.99,
    algorithm="td"
)

# Update values
rl.update_values(
    state_key="state_1",
    reward=1.0,
    next_state_key="state_2"
)

# Get value estimate
value = rl.get_value("state_1")
```

### 2. Q-Learning

Off-policy algorithm that learns optimal action-value function.

```python
# Initialize with Q-learning
rl = ReinforcementLearningIntegrator(
    learning_rate=0.1,
    discount_factor=0.95,
    algorithm="qlearning"
)

# Update Q-value
td_error = rl.update_q_value(
    state_key="state_1",
    action=0,
    reward=1.0,
    next_state_key="state_2",
    done=False
)

# Select action (epsilon-greedy)
action = rl.select_action("state_1", num_actions=4, exploration=True)

# Get Q-value
q_val = rl.get_q_value("state_1", action=0)
```

### 3. Policy Gradient (REINFORCE)

Policy-based method that directly optimizes the policy.

```python
# Initialize with policy gradient
rl = ReinforcementLearningIntegrator(
    learning_rate=0.001,
    discount_factor=0.99,
    algorithm="policy_gradient"
)

# Update policy
rl.update_policy_gradient(
    state_key="state_1",
    action=2,
    reward=1.5,
    state_features=state_vector  # Optional feature representation
)
```

### 4. Actor-Critic

Combines value-based and policy-based methods for stable learning.

```python
# Initialize with actor-critic
rl = ReinforcementLearningIntegrator(
    learning_rate=0.01,
    discount_factor=0.99,
    algorithm="actor_critic"
)

# Update both actor and critic
critic_loss, actor_loss = rl.update_actor_critic(
    state_key="state_1",
    action=1,
    reward=0.5,
    next_state_key="state_2",
    state_features=state_vector,
    done=False
)
```

## Integration with Neural Networks

### Motor Control Example

```python
from src.motor_output import ReinforcementLearningIntegrator, extract_motor_commands
from src.brain_model import BrainModel

# Create brain model
model = BrainModel(lattice_size=(10, 10, 10, 5))

# Initialize RL integrator
rl_integrator = ReinforcementLearningIntegrator(
    learning_rate=0.01,
    algorithm="qlearning"
)

# Training loop
for episode in range(1000):
    state_key = "start"
    total_reward = 0
    
    for step in range(100):
        # Get motor commands from neural network
        motor_output = extract_motor_commands(model, num_actions=4)
        
        # Select action based on neural output and Q-values
        action = rl_integrator.select_action(state_key, num_actions=4)
        
        # Execute action in environment (user-defined)
        next_state_key, reward, done = execute_action(action)
        
        # Update Q-values
        rl_integrator.update_q_value(
            state_key, action, reward, next_state_key, done
        )
        
        # Step neural network
        model.step()
        
        total_reward += reward
        state_key = next_state_key
        
        if done:
            break
    
    print(f"Episode {episode}: Total Reward = {total_reward}")
```

## See Also

- [Model Comparison Tools](MODEL_COMPARISON.md)
- [Information Theory Metrics](INFORMATION_THEORY.md)
- [Tasks and Evaluation](../tutorials/TASKS_EVALUATION.md)
