import tensorflow as tf


from evaluation.multi_class_evaluation import MultiClass_Evaluator



class Utils:
    def __init__(self):
        pass

    @staticmethod
    def get_last_layer_activation(problem,type):
        if type=="multi_class":
            return "softmax"


    @staticmethod
    def get_initializer(
            initializer="uniform",
            mean=0.0,
            stddev=0.05,
            minval=-0.05,
            maxval=0.05,
            gain=1.0,
            mode="fan_in",
            constant_value=0,
            variance_scale=1.0,
            distribution="truncated_normal"
    ):
        if initializer=="random_normal":
            return tf.keras.initializers.RandomNormal(
                    mean=mean,
                    stddev=stddev,
                    seed=None
            )

        elif initializer=="random_uniform":
            return tf.keras.initializers.RandomUniform(
                    minval=minval,
                    maxval=maxval,
                    seed=None
            )
        if initializer=="truncated_normal":
            return tf.keras.initializers.TruncatedNormal(
                    mean=mean,
                    stddev=stddev,
                    seed=None
            )

        elif initializer=="zeros":
            return tf.keras.initializers.Zeros()

        elif initializer=="ones":
            return tf.keras.initializers.Ones()

        elif initializer=="glorot_normal":
            return tf.keras.initializers.GlorotNormal(seed=None)

        elif initializer=="glorot_uniform":
            return tf.keras.initializers.GlorotUniform(seed=None)

        elif initializer=="identity":
            return tf.keras.initializers.Identity(gain=gain)

        elif initializer=="orthogonal":
            return tf.keras.initializers.Orthogonal(gain=gain, seed=None)

        elif initializer=="constant":
            return tf.keras.initializers.Constant(value=constant_value)

        elif initializer=="":
            return tf.keras.initializers.VarianceScaling(
                    scale=variance_scale,
                    mode=mode,
                    distribution=distribution,
                    seed=None
            )

    @staticmethod
    def get_regularizer(
            regularizer="l1",
            l1=0.01,
            l2=0.01,
    ):
        if regularizer=="l1":
            return tf.keras.regularizers.l1(l1)

        elif regularizer=="l2":
            return tf.keras.regularizers.l2(l2)

        elif regularizer=="l1_l2":
            return tf.keras.regularizers.l1_l2(l1,l2)

    @staticmethod
    def get_evaluator(config):
        if config.type=="multi_class":
            return MultiClass_Evaluator(config)







