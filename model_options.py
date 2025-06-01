import cnst


class ModelOptions:
    def __init__(self):
        self.embedding_dim = 64

        self.neg_sampling_opt = cnst.UNIFORM

        self.adversarial_temp = 1.0
        self.nb_neg_examples_per_pos = 25
        self.loss_margin = 3.0
        self.loss_fct = cnst.POLY_V2_LOSS
        self.obj_fct = cnst.NEG_SAMP
        self.batch_size = 1024
        self.loss_norm_ord = 2
        self.regularisation_lambda = 0
        self.total_log_box_size = -5
        self.regularisation_points = 0
        self.hard_total_size = False
        self.hard_code_size = False

        self.learning_rate = 1e-3
        self.stop_gradient = cnst.NO_STOPS

        self.restricted_training = False
        self.restriction = 1024

        self.replace_indices = False

        self.shared_shape = False # Don't change
        self.learnable_shape = True # Don't change also
        self.fixed_width = False

        self.bounded_pt_space = False
        self.bounded_box_space = False
        self.space_bound = 1.0

        self.dim_dropout_prob = 0.0
        self.gradient_clip = -1.0

        self.lambda_neg_reconstruction = 0.2

        self.bounded_norm = True
        self.learning_rate_decay = 0
        self.decay_period = 100

        self.augment_inv = False
        self.generate_report = False

        self.viz = False
        self.pos = False
        self.verbose = False

        self.typ_thresh = 0.8
