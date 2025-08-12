from alignment.utils.drgrpo_grader import r1_zero_reward_fn

def make_reward_fn(ground_truths):
    def reward_fn(output_text, idx):
        ref = ground_truths[idx]
        return r1_zero_reward_fn(output_text, ref)
    return reward_fn