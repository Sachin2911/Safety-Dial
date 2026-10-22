"""Keep policy sampling reproducible across separately identified study episodes."""


class EpisodeSeededPolicy:
    def __init__(self, policy, key_factory):
        self.policy = policy
        self.key_factory = key_factory
        self.last_episode = None

    def infer(self, observation):
        observation = dict(observation)
        episode = observation.pop("study_episode_id")
        seed = int(observation.pop("episode_seed"))
        identity = (episode, seed)
        if identity != self.last_episode:
            self.policy._rng = self.key_factory(seed)
            self.last_episode = identity
        return self.policy.infer(observation)
