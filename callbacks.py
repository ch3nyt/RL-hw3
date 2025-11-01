from stable_baselines3.common.callbacks import BaseCallback

class BucketCollector(BaseCallback):
    def __init__(self, bucket, **kwargs):
        super().__init__(**kwargs)
        self.bucket = bucket

    def _on_step(self) -> bool:
        """
        這是 BaseCallback 要求必須實作的抽象方法。
        我們不需要在每一步都做任何事，所以
        只需要回傳 True 讓訓練繼續即可。
        """
        return True

    def _on_rollout_end(self) -> bool:
        """
        在每一次 rollout 結束後 (收集完 n_steps * num_envs 筆資料後)，
        從所有環境中收集最終的盤面。
        """
        try:
            # 從 VecEnv 取每個子環境的盤面
            boards = self.training_env.env_method("get_board")
            for bd in boards:
                self.bucket.push(bd)
        except Exception as e:
            # 增加一個錯誤處理，避免因為某個環境出錯導致訓練崩潰
            print(f"[BucketCollector] Error collecting boards: {e}")
            
        return True