class GlobalConfig:
    """
    Attributes
    ----------

    jit_module:
        whether complete flax module should be compiled using lifted jit (`nn.jit`), default=False.

    unroll_rnn:
        how many scan iterations to unroll within a single iteration of a loop,
        defaults to 1. This argument will be passed to `nn.OptimizedLSTMCell(unroll=...)`.
    """

    jit_module: bool = False
    unroll_rnn: int = 1

    @classmethod
    def update(cls, jit_module: bool | None = None, unroll_lstm: int | None = None):
        cls.jit_module = jit_module or cls.jit_module
        cls.unroll_rnn = unroll_lstm or cls.unroll_rnn

    @classmethod
    def get(cls):
        return GlobalConfig()
