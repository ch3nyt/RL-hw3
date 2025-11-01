import warnings
import time
import gymnasium as gym
from gymnasium.envs.registration import register

# import wandb
from wandb.integration.sb3 import WandbCallback
# --- ↓↓↓ 新增這些 import ↓↓↓ ---
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- ↑↑↑ 新增這些 import ↑↑↑ ---
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

from carousel import ResetWithBucket, STATE_BUCKET, Log2Wrapper
from callbacks import BucketCollector
from torch import nn

warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

class CustomCNN(BaseFeaturesExtractor):
    """
    用於 2048 環境 (16, 4, 4) 觀測空間的自訂 CNN。
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # features_dim 是 CNN 最終輸出的特徵維度
        super().__init__(observation_space, features_dim)
        
        # 我們的觀測空間是 (16, 4, 4)，所以 n_input_channels 是 16
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # 輸入: (N, 16, 4, 4)
            nn.Conv2d(n_input_channels, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # 輸出: (N, 64, 3, 3)
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            # 輸出: (N, 128, 2, 2)
            nn.Flatten(), # 輸出: (N, 128 * 2 * 2) = (N, 512)
        )

        # 建立一個線性層，將 CNN 的輸出 (512) 映射到 PPO 需要的 features_dim
        # (預設 512 -> 512)
        self.linear = nn.Sequential(
            nn.Linear(128 * 2 * 2, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 錯誤訊息顯示 env 傳來 int64，但 CNN 需要 float
        return self.linear(self.cnn(observations.float()))
    
policy_kwargs = dict(
    features_extractor_class=CustomCNN, # ★ 告訴 PPO 使用我們的 CNN
    features_extractor_kwargs=dict(features_dim=512), # ★ CNN 輸出的維度
    
    net_arch=[256, 256, 256],  # CNN(512) -> 256 -> 256 -> 256
    activation_fn=nn.ReLU,
)
# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",
    "algorithm": PPO,
    "policy_network": "CnnPolicy",
    "save_path": "models/sample_model",
    "num_train_envs": 24,
    "epoch_num": 200,
    "timesteps_per_epoch": 81920*5,
    "eval_episode_num": 30,
}

def make_env():
    env = gym.make('2048-v0')
    env = ResetWithBucket(env, STATE_BUCKET) 
    # env = Log2Wrapper(env) # ★ Log2Wrapper 必須在 ResetWithBucket 之後
    return env
    
def make_env_train():
        return make_env()

def make_env_eval():
    """建立乾淨的評估環境"""
    env = gym.make('2048-v0')
    # (不需要 Log2Wrapper，也不需要 ResetWithBucket)
    return env

def eval(env, model, eval_episode_num):
    """Evaluate the model and return avg_score and avg_highest"""
    avg_score = 0
    avg_highest = 0
    for seed in range(eval_episode_num):
        
        # --- 使用 VecEnv API ---
        # 1. VecEnv 使用 .seed() 方法設定種子
        env.seed(seed)
        # 2. VecEnv.reset() 不接受參數，且只回傳 obs
        obs = env.reset()
        # 3. done 在 VecEnv 中是一個陣列，例如 [False]
        done = [False]
        # ------------------------

        # Interact with env
        # 4. 檢查 done[0] (因為 eval_env 只有一個環境)
        while not done[0]:
            action, _state = model.predict(obs, deterministic=True)
            
            # --- 使用 VecEnv API ---
            # 5. VecEnv.step() 回傳 (obs, reward, done, info)
            #    它沒有 terminated 和 truncated
            obs, reward, done, info = env.step(action)
            # ------------------------
        
        # info[0] 是因為 eval_env 也是 VecEnv (size=1)
        avg_highest += info[0]['highest']
        avg_score   += info[0]['score']

    avg_highest /= eval_episode_num
    avg_score /= eval_episode_num
        
    return avg_score, avg_highest


def train(eval_env, model, config):
    """Train agent using SB3 algorithm and my_config"""
    current_best_score = 0
    current_best_highest = 0

    start_time = time.time()

    for epoch in range(config["epoch_num"]):
        epoch_start_time = time.time()

        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            callback=BucketCollector(STATE_BUCKET),   # ★ 每輪 rollout 後補貨到桶
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        epoch_duration = time.time() - epoch_start_time
        total_duration = time.time() - start_time

        ### Evaluation
        eval_start = time.time()
        avg_score, avg_highest = eval(eval_env, model, config["eval_episode_num"])
        eval_duration = time.time() - eval_start

        # Print training progress and speed
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epoch_num']} completed")
        print(f"{'='*60}")
        print(f"Training Speed:")
        print(f"   - Epoch time: {epoch_duration:.1f}s")
        print(f"   - Eval time: {eval_duration:.1f}s")
        print(f"   - Total time: {total_duration/60:.1f} min")
        print(f"Performance:")
        print(f"   - Avg Score: {avg_score:.1f}")
        print(f"   - Avg Highest Tile: {avg_highest:.1f}")


        # wandb.log(
        #     {"avg_highest": avg_highest,
        #      "avg_score": avg_score}
        # )
        

        ### Save best model
        if current_best_score < avg_score or current_best_highest < avg_highest:
            print("Saving New Best Model")
            if current_best_score < avg_score:
                print(f"   - Previous best score: {current_best_score:.1f} → {avg_score:.1f}")
                current_best_score = avg_score
            if current_best_highest < avg_highest:
                print(f"   - Previous best tile: {current_best_highest:.1f} → {avg_highest:.1f}")
                current_best_highest = avg_highest

            save_path = config["save_path"]
            model.save(f"{save_path}/best")

        print("-"*60)
            
    total_time = (time.time() - start_time)
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")

if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization  # NEW

    # --- build raw envs as before ---
    # 3. 將 DummyVecEnv 改為 SubprocVecEnv
    train_env = SubprocVecEnv([make_env_train for _ in range(my_config["num_train_envs"])])    
    eval_env  = DummyVecEnv([make_env_eval])

    # 訓練環境：正規化 Reward
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # 評估環境：不同步 Reward
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    
    def linear_schedule(start, end=0.0):
        return lambda progress_remaining: end + (start - end) * progress_remaining
    
    model = my_config["algorithm"](
        my_config["policy_network"], 
        train_env,
        # n_epochs=15,
        # target_kl=0.01,         # auto-stop overly large steps
        n_steps=2048*5,           # rollout = 2048 * 8 = 16384
        batch_size=8192,        # 2 minibatches
        n_epochs=10,            # was 12; reduce to avoid hitting KL too fast
        learning_rate=linear_schedule(3e-4, 3e-6),
        clip_range=linear_schedule(0.2, 0.05),
        ent_coef=0.05,
        verbose=1,
        policy_kwargs=policy_kwargs,   # ⬅️ add this
        tensorboard_log=my_config["run_id"],
    )

    train(eval_env, model, my_config)
