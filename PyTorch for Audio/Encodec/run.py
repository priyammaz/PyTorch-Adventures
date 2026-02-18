import yaml
import argparse
from modules.encodec import EncodecModel, EnCodecConfig
from modules.discriminator import Discriminator, DisciminatorConfig
from trainer import EncodecTrainer

def load_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_yaml(args.path_to_config)
    
    ### Get Training Config ###
    training_config = config["training_config"]

    ### Get Generator Config ###
    generator_config = config["generator_config"]
    
    ### Get Discriminator Config and in_channels from Generator ###
    discriminator_config = config["discriminator_config"]
    discriminator_config["in_channels"] = discriminator_config["out_channels"] = generator_config["channels"]
    
    ### Load Encodec Model ###
    encodec_config = EnCodecConfig(**generator_config)
    encodec = EncodecModel(encodec_config)
 
    ### Load Discriminators ###
    use_disc = any([discriminator_config["use_multiscale_freq_discrim"], \
                    discriminator_config["use_multiscale_freq_discrim"], \
                    discriminator_config["use_multiscale_freq_discrim"]])
    if use_disc:
        disc_config = DisciminatorConfig(**discriminator_config)
        discriminator = Discriminator(disc_config)
    else:
        discriminator = None

    trainer = EncodecTrainer(generator=encodec, 
                             discriminator=discriminator, 
                             training_config=training_config)
    trainer.train()

if __name__ == "__main__":
    main()
    
