import model.actor_critic
import torch

from ray.rllib.models import ModelCatalog


def load_torch_model(import_path, action_space, observation_space, config):
    # This model needs to be the specific model used by RLlib. Check config if we're in the ALE namespace, then use a VisionNet and use RLlib's model creator.
    if not config["custom_model"]:
        _, logit_dim = ModelCatalog.get_action_dist(
            action_space, config, framework="torch"
        )

        loaded_model = ModelCatalog.get_model_v2(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=logit_dim,
            model_config=config,
            framework="torch",
        )
    elif config["custom_model"] == "actor_critic":
        loaded_model = model.actor_critic.ActorCritic(observation_space, action_space, action_space.n, config, "ActorCritic")
    else:
        raise("Unknown model type.")

    loaded_model.load_state_dict(torch.load(import_path, map_location=next(loaded_model.parameters()).device))
    loaded_model.eval()

    return loaded_model


def register_models():
    ModelCatalog.register_custom_model("actor_critic", model.actor_critic.ActorCritic)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)