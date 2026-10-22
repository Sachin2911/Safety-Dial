"""Keep independent sampling streams for interleaved local simulator clients."""


class IsolatedEpisodePolicy:
    def __init__(self, policy, key_factory):
        self.policy = policy
        self.key_factory = key_factory
        self.episodes = {}

    def infer(self, observation):
        observation = dict(observation)
        episode = observation.pop('study_episode_id')
        seed = int(observation.pop('episode_seed'))
        if episode in self.episodes:
            prior_seed, key = self.episodes[episode]
            if prior_seed != seed:
                raise ValueError('A study episode cannot change its sampling seed')
        else:
            key = self.key_factory(seed)
        self.policy._rng = key
        try:
            return self.policy.infer(observation)
        finally:
            self.episodes[episode] = (seed, self.policy._rng)
