import jax 
import jax.numpy as jnp

class parametric_model:
    """
    class for family of velocity models with linear combination of basis functions
    """

    def __init__(self, basis_name = "linear", add_exponential_terms = False, add_fourier_terms = False):
        assert basis_name in ["linear", "quadratic", "cubic", "quartic"], "basis_name must be either 'linear' or 'quadratic'."
        self.basis_name = basis_name
        self.add_exponential_terms = add_exponential_terms
        self.add_fourier_terms = add_fourier_terms
        if basis_name == "linear":
            self.nparams = 3
        elif basis_name == "quadratic":
            self.nparams = 6
        elif basis_name == "cubic":
            self.nparams = 10 
        elif basis_name == "quartic":
            self.nparams = 15 
        if add_exponential_terms:
            self.nparams += 3
        if add_fourier_terms:
            self.nparams += 4

    def basis(self, y, t):
        if y.ndim == 0:
            y = jnp.expand_dims(y, (0,1))
            t = jnp.expand_dims(t, (0,1))
        if y.ndim == 1:
            y = jnp.expand_dims(y, -1)
            t = jnp.expand_dims(t, -1)
        if self.basis_name == "linear":
            basis = jnp.hstack((jnp.ones_like(t), t, y))
        elif self.basis_name == "quadratic":
            basis = jnp.hstack((jnp.ones_like(t), t, y, t**2, y**2, t*y))
        elif self.basis_name == "cubic":
            basis = jnp.hstack((jnp.ones_like(t), t, y, t**2, y**2, t*y, t**3, y**3, t**2*y, t*y**2))
        elif self.basis_name == "quartic":
            basis = jnp.hstack((jnp.ones_like(t), t, y, t**2, y**2, t*y, t**3, y**3, t**2*y, t*y**2, t**4, y**4, t**3*y, t**2*y**2, t * y**3))
        if self.add_exponential_terms:
            basis = jnp.hstack((basis, jnp.exp(-t**2), jnp.exp(-y**2), jnp.exp(-t**2 - y**2)))
        if self.add_fourier_terms:
            basis = jnp.hstack((basis, jnp.sin(t), jnp.cos(t), jnp.sin(y), jnp.cos(y)))
        return basis

    def __call__(self, y, t, params):
        return jnp.dot(self.basis(y, t), params["params_l2"])
    
    def params_init(self, seed = 0, init_weight = 1.0):
        p_init = {
            'params_l2': jax.random.normal(jax.random.PRNGKey(seed), (self.nparams,)) * init_weight
        }
        return p_init
    
class nn_model:
    """
    base class for neural network velcoity models 
    """
    def __init__(self, layers = 2, hidden_size = 32):
        self.layers = layers
        self.hidden_size = hidden_size
    
    def forward(self, y, t, params):
        x = jnp.stack((y,t), axis = -1)
        if self.layers >1 :
            h = x
            for i in (range(self.layers-1)):
                weight = params[f"w{i+1}_l2"]
                h = jnp.dot(h, weight) + params[f"b{i+1}"]
                h = jax.nn.relu(h)
            out = jnp.dot(h, params[f"w{self.layers}_l2"]) + params[f"b{self.layers}"]
        else:
            out = jnp.dot(x, params["w1_l2"]) + params["b1"]
        return out
    
    def __call__(self, y, t, params):
        out = self.forward(y, t, params)
        out = out.reshape(-1,)
        return out
    
    def params_init(self, seed, init_weight = 1.0, in_features = 2):
        p_init = {'w1_l2': jax.random.normal(shape=(in_features, self.hidden_size), key=jax.random.PRNGKey(seed + 0)) * init_weight,
                'b1': jax.random.normal(shape=(self.hidden_size), key=jax.random.PRNGKey(seed + 1)) * init_weight}

        seed = seed + 2
        
        if self.layers > 1:

            for i in range(self.layers-2):
                p_init[f'w{i+2}_l2'] = jax.random.normal(shape=(self.hidden_size, self.hidden_size), key=jax.random.PRNGKey(seed)) * init_weight
                p_init[f'b{i+2}'] = jax.random.normal(shape=(self.hidden_size), key=jax.random.PRNGKey(seed + 1)) * init_weight
                seed = seed + 2

            p_init[f'w{self.layers}_l2'] = jax.random.normal(shape=(self.hidden_size, 1), key=jax.random.PRNGKey(seed)) * init_weight
            p_init[f'b{self.layers}'] = jax.random.normal(shape=(), key=jax.random.PRNGKey(seed + 1)) * init_weight

            seed = seed + 2

        return p_init


class anm_model(nn_model):
    """
    velocity implementation of anm (v(y,x) = f(x))
    """
    def forward(self, y, t, params):
        x = jnp.stack((t,t), axis = -1)
        h = x
        if self.layers >1 :
            for i in (range(self.layers-1)):
                weight = params[f"w{i+1}_l2"]
                h = jnp.dot(h, weight) + params[f"b{i+1}"]
                h = jax.nn.relu(h)
            out = jnp.dot(h, params[f"w{self.layers}_l2"]) + params[f"b{self.layers}"]
        else:
            out = jnp.dot(x, params["w1_l2"]) + params["b1"]
        return out
    
    def __call__(self, y, t, params):
        m = self.forward(y, t, params)
        m = m.reshape(-1,)
        return m

class lsnm_model(nn_model):
    """
    velocity implementation of lsnm 
    """
    def params_init(self, seed, init_weight=1):
        p = {}
        p["offset"] = super().params_init(seed, init_weight, in_features = 1)
        p["scale"] = super().params_init(seed + 444, init_weight, in_features = 1)
        return p
    
    def forward(self, x, params):
        h = x.reshape(-1,1)
        if self.layers >1 :
            for i in (range(self.layers-1)):
                weight = params[f"w{i+1}_l2"]
                h = jnp.dot(h, weight) + params[f"b{i+1}"]
                h = jax.nn.tanh(h)
            out = jnp.dot(h, params[f"w{self.layers}_l2"]) + params[f"b{self.layers}"]
        else:
            out = jnp.dot(x, params["w1_l2"]) + params["b1"]
        return out
    
    def __call__(self, y, t, params):
        m = lambda x: self.forward(x, params["offset"])
        s = lambda x: self.forward(x, params["scale"])

        m_, dm_ = jax.jvp(m, (t,), (jnp.ones_like(t),))
        s_, ds_ = jax.jvp(s, (t,), (jnp.ones_like(t),)) ## assume s is exp(h), then d log s = dh. 

        m_, dm_, s_, ds_ = m_.reshape(-1,), dm_.reshape(-1,), s_.reshape(-1,), ds_.reshape(-1,)

        return dm_ + ds_ * ((y - m_)) 

class additive_model(nn_model):
    """
    velocity model of the form v(y,x) = f(x) + g(y)
    """
    def params_init(self, seed, init_weight=1):
        p = {}
        p["f"] = super().params_init(seed, init_weight, in_features = 1)
        p["g"] = super().params_init(seed + 444, init_weight, in_features = 1)
        return p
    
    def forward(self, x, params):
        if self.layers >1 :
            h = x.reshape(-1,1)
            for i in (range(self.layers-1)):
                weight = params[f"w{i+1}_l2"]
                h = jnp.dot(h, weight) + params[f"b{i+1}"]
                h = jax.nn.relu(h)
            out = jnp.dot(h, params[f"w{self.layers}_l2"]) + params[f"b{self.layers}"]
        else:
            out = jnp.dot(x, params["w1_l2"]) + params["b1"]
        return out
    
    def __call__(self, y, t, params):
        f = self.forward(t, params["f"]).reshape(-1,)
        g = self.forward(y, params["g"]).reshape(-1,)
        return f + g
