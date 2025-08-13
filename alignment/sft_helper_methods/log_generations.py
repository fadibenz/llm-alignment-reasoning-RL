import logging
from typing import List


def log_generations(samples: List[str], logger: logging.Logger):

    logger.info("\n=== Sample Generations ===")

    for i, (prompt, response, ground_truth, answer_reward, format_reward, length, entropy) in enumerate(samples, 1):
        status = "✓" if answer_reward == 1 else "✗"
        logger.info(f"\n--- Sample {i} {status} ---")
        logger.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        logger.info(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        logger.info(f"Ground Truth: {ground_truth}")
        logger.info(f"Rewards: Answer={answer_reward}, Format={format_reward} | Length={length}, Entropy={entropy:.3f}")