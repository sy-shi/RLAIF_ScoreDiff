import model.util
import numpy as np
import pickle
import torch
import torch.nn.functional as F

from ray.rllib.utils.torch_utils import convert_to_torch_tensor

from enum import Enum


class ModelType(Enum):
    TORCH = 1
    TREE = 2


class ModelWrapper():
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def load(self, import_path, action_space, observation_space, config):
        if self.model_type == ModelType.TORCH:
            self.model = model.util.load_torch_model(import_path, action_space, observation_space, config)
        elif self.model_type == ModelType.TREE:
            self.model = pickle.load(open(import_path, 'rb'))

    def set(self, in_model):
        self.model = in_model

    def get_action(self, obs):
        obs = self._preprocess_obs(obs)

        if self.model_type == ModelType.TORCH:
            action_logit = self.model({"obs": obs})[0]
            action_prob = F.softmax(action_logit, dim=1).cpu().detach().numpy()
            log_action_prob = np.log(action_prob)
            action = np.argmax(action_prob)

            # Use maximum entropy formulation to estimate q values
            importance = np.max(log_action_prob) - np.min(log_action_prob)
        elif self.model_type == ModelType.TREE:
            action = self.model.predict([obs])[0]
            action_prob = self.model.predict_proba([obs])[0]
            importance = None

        return action, action_prob, importance

    def get_explanation(self, obs, action):
        explanation = None

        obs = self._preprocess_obs(obs)

        if self.model_type == ModelType.TREE:
            feature = self.model.tree_.feature
            threshold = self.model.tree_.threshold

            node_indicator = self.model.decision_path([obs])
            leaf_id = self.model.apply([obs])[0]

            # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
            node_index = node_indicator.indices[
                node_indicator.indptr[0] : node_indicator.indptr[1]
            ]

            explanation = []
            for node_id in node_index:
                # continue to the next node if it is a leaf node
                if leaf_id == node_id:
                    continue

                # check if value of the split feature for sample 0 is below threshold
                if obs[feature[node_id]] <= threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                explanation.append({'node': node_id,
                            'feature': feature[node_id],
                            'value': obs[feature[node_id]],
                            'inequality': threshold_sign,
                            'threshold': threshold[node_id],
                            'is_leaf': False})

            explanation.append({'node': leaf_id,
                        'value': action,
                        'is_leaf': True})

        return explanation

    def _preprocess_obs(self, obs):
        if self.model_type == ModelType.TORCH:
            # if type(obs) != torch.Tensor:
            #     obs = torch.tensor(obs, requires_grad = False)
            obs = convert_to_torch_tensor(obs, device=next(self.model.parameters()).device)

            # Need to add batch dim.
            for key in obs.keys():
                obs[key] = torch.unsqueeze(obs[key], 0)

        elif self.model_type == ModelType.TREE:
            if type(obs) == torch.Tensor:
                obs = obs.cpu().detach().numpy()

            # Flatten observation if it is multiple dimensions (excluding batch)
            if len(obs.shape) > 2:
                obs = obs.flatten()

        return obs