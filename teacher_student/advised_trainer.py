def get_advised_trainer(base_class, policy_class):

    class AdvisedTrainer(base_class):
        def get_default_policy_class(self, config):
            return policy_class

    return AdvisedTrainer