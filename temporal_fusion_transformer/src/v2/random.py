import keras_core.backend
from keras_core.backend import backend

if keras_core.backend.backend() == "jax":
    
    import jax.random

    class SeedGenerator:
    
        def __init__(self, prng_seed: int = 42):
            self.prng_key = jax.random.PRNGKey(prng_seed)
            
            jax.random.fold_in()
        
        def draw_seed(self):
            (self.prng_key, new_seed) = jax.random.split(self.prng_key, 2)