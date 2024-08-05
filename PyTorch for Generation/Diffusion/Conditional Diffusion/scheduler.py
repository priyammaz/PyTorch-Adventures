import torch

###################################################
################ DEFINE SAMPLER ###################
###################################################

### DEFINE DDPM SAMPLER ###
class Sampler:
    def __init__(self, total_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.total_timesteps = total_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        ### Define Basic Beta Scheduler ###
        self.beta_schedule = self.linear_beta_schedule()

        ### Compute Alphas for Direction 0 > t Noise Calculation ###
        self.alpha = 1 - self.beta_schedule
        self.alpha_cumulative_prod = torch.cumprod(self.alpha, dim=-1)
    
    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.total_timesteps)

    def _repeated_unsqueeze(self, target_shape, input):
        while target_shape.dim() > input.dim():
            input = input.unsqueeze(-1)
        return input
    
    def add_noise(self, inputs, timesteps):

        batch_size, c, h, w = inputs.shape

        ### Grab the Device we want to place tensors on ###
        device = inputs.device
        
        alpha_cumulative_prod_timesteps = self.alpha_cumulative_prod[timesteps].to(device)
        
        ### Compute Mean Coefficient ###
        mean_coeff = alpha_cumulative_prod_timesteps ** 0.5

        ### Compute Variance Coefficient ###
        var_coeff = (1 - alpha_cumulative_prod_timesteps) ** 0.5

        ### Reshape mean_coeff and var_coeff to have shape (batch x 1 x 1 x 1) so we can broadcast with input (batch x c x height x width) ###
        mean_coeff = self._repeated_unsqueeze(inputs, mean_coeff)
        var_coeff = self._repeated_unsqueeze(inputs, var_coeff)

        ### Generate some Noise X ~ N(0,1) (rand_like will automatically place on same device as the inputs) ###
        noise = torch.randn_like(inputs)
        
        ### Compute Mean (mean_coef * x_0) ###
        mean = mean_coeff * inputs

        ### Compute Variance ###
        var = var_coeff * noise

        ### Compute Noisy Data ###
        noisy_image = mean + var

        return noisy_image, noise
        
    def remove_noise(self, input, timestep, predicted_noise):

        assert (input.shape == predicted_noise.shape), "Shapes of noise pattern and input image must be identical!!"
        
        b, c, h, w = input.shape

        ### Grab Device to Place Tensors On ###
        device = input.device

        ### Create a mask (if timestep == 0 sigma_z will also be 0 so we need to save this for later ###
        greater_than_0_mask = (timestep >= 1).int()

        
        ### Compute Sigma (b_t * (1 - cumulative_a_(t-1)) / (1 - cumulative_a)) * noise ###
        alpha_cumulative_t = self.alpha_cumulative_prod[timestep].to(device)
        alpha_cumulative_prod_t_prev = self.alpha_cumulative_prod[timestep - 1].to(device) # (timestep - 1) if timestep is 0 is WRONG! we will multiply by 0 later
        beta_t = self.beta_schedule[timestep].to(device)
        noise = torch.randn_like(input)
        variance = beta_t * (1 - alpha_cumulative_prod_t_prev) / (1 - alpha_cumulative_t)

        ### 0 out the variance for if the timestep == 0 ###
        variance = variance * greater_than_0_mask
        variance = self._repeated_unsqueeze(input, variance)
        sigma_z = noise * variance**0.5

        ### Compute Noise Coefficient (1 - a_t / sqrt(1 - cumulative_a)) where 1 - a_t = b_t ###
        beta_t = self.beta_schedule[timestep].to(device)
        alpha_cumulative_t = self.alpha_cumulative_prod[timestep].to(device)
        root_one_minus_cumulative_alpha_t = (1 - alpha_cumulative_t) ** 0.5
        noise_coefficient = beta_t / root_one_minus_cumulative_alpha_t
        noise_coefficient = self._repeated_unsqueeze(input, noise_coefficient)
        

        ### Compute 1 / sqrt(a_t) ###
        reciprocal_root_a_t = (self.alpha[timestep]**-0.5).to(device)
        reciprocal_root_a_t = self._repeated_unsqueeze(input, reciprocal_root_a_t)
        
        ### Compute Denoised Image ###
        denoised = reciprocal_root_a_t * (input - (noise_coefficient * predicted_noise)) + sigma_z
 
        return denoised