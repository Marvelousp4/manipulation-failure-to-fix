"""Stream selected episodes to MP4 without retaining frames in memory."""

from failure2fix.policies import as_numpy


class RolloutVideos:
    def __init__(self, directory, seeds, enabled=True):
        self.directory = directory
        self.seeds = set(seeds) if enabled else set()
        self.key = None
        self.writer = None

    def __call__(self, env, condition, seed, step):
        if seed not in self.seeds:
            return
        key = (condition, seed)
        if key != self.key:
            self.close()
            import imageio.v2 as imageio

            self.directory.mkdir(parents=True, exist_ok=True)
            fps = env.unwrapped.control_freq
            self.writer = imageio.get_writer(self.directory / f"{condition}-{seed}.mp4", fps=fps)
            self.key = key
        frame = as_numpy(env.render())
        if frame.ndim == 4:
            frame = frame[0]
        self.writer.append_data(frame)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.key = None
