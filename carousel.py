# carousel.py
import numpy as np, gymnasium as gym
from itertools import cycle

BUCKET_ORDER = [(512, None), (256, None), (128, None), (None, 10)]

class StateBucket:
    def __init__(self):
        self.bank = {b: [] for b in BUCKET_ORDER}
    def push(self, board: np.ndarray):
        maxv = int(board.max()); empty = int((board==0).sum())
        if maxv >= 512: self.bank[(512,None)].append(board.copy())
        elif maxv >= 256: self.bank[(256,None)].append(board.copy())
        elif maxv >= 128: self.bank[(128,None)].append(board.copy())
        if empty >= 10: self.bank[(None,10)].append(board.copy())
    def sample(self, bucket):
        arr = self.bank[bucket]
        return None if not arr else arr[np.random.randint(len(arr))]

STATE_BUCKET = StateBucket()

class ResetWithBucket(gym.Wrapper):
    def __init__(self, env, bucket, order=BUCKET_ORDER):
        super().__init__(env)
        self.bucket = bucket
        self._order = cycle(order)
    def reset(self, **kwargs):
        # 依序嘗試不同桶，找到就塞進去
        for _ in range(len(BUCKET_ORDER)):
            b = next(self._order)
            bd = self.bucket.sample(b)
            if bd is not None:
                # 透過 .unwrapped 繞過所有 Wrapper，直接呼叫 My2048Env 的方法
                self.env.unwrapped.set_pending_start_board(bd) # <--- 修正後
                break
        return self.env.reset(**kwargs)

class Log2Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 更新觀測空間（可選，但好習慣）
        # 假設原始空間是 Box(0, 2048, (4,4), np.int32)
        # log2(2048) = 11. log2(0) 處理為 0.
        self.observation_space = gym.spaces.Box(
            low=0, high=16, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        # np.log2(0) 會是 -inf，所以我們先 +1
        # obs_plus_one = obs + 1
        # log_obs = np.log2(obs_plus_one)
        
        # 一個更安全且常用的方法：
        obs = obs.astype(np.float32)
        obs[obs > 0] = np.log2(obs[obs > 0])
        return obs