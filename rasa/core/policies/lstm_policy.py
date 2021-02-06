import copy
import json
import logging
import os
import tensorflow as tf
import numpy as np
import warnings
from typing import Any, List, Dict, Text, Optional, Tuple, Union

from pathlib import Path
import rasa.shared.utils.io

from rasa.shared.core.domain import Domain
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer)
from rasa.core.featurizers.single_state_featurizer import BinarySingleStateFeaturizer
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
import rasa.utils.common as common_utils
from rasa.core.constants import DEFAULT_POLICY_PRIORITY
#new
from rasa.utils.tensorflow.model_data import Data
import rasa.utils.tensorflow.model_data_utils as model_data_utils
from rasa.shared.nlu.constants import ACTION_TEXT, TEXT
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from collections import defaultdict, OrderedDict
from rasa.utils.tensorflow.constants import SENTENCE
import scipy.sparse
from sklearn.utils import shuffle as sklearn_shuffle

# there are a number of issues with imports from tensorflow. hence the deactivation
# pytype: disable=import-error
# pytype: disable=module-attr


try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)


class LSTMPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    defaults = {
        # Neural Net and training params
        "rnn_size": 32,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.1,
        # set random seed to any int to get reproducible results
        "random_seed": None,
    }

    @staticmethod
    def _standard_featurizer(max_history=None) -> MaxHistoryTrackerFeaturizer:
        return MaxHistoryTrackerFeaturizer(
            BinarySingleStateFeaturizer(), max_history=max_history
        )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        model: Optional[tf.keras.models.Sequential] = None,
        current_epoch: int = 0,
        max_history: Optional[int] = None,
        label_encoder: LabelEncoder = LabelEncoder(),
        # shuffle: bool = True,
        zero_state_features: Optional[Dict[Text, List["Features"]]] = None,
        **kwargs: Any,
    ) -> None:
        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        super().__init__(featurizer, priority)

        self._load_params(**kwargs)
        self.model = model

        #new add
        self.label_encoder = label_encoder
        self.zero_state_features = zero_state_features or defaultdict(list)

        self.current_epoch = current_epoch

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        # filter out kwargs that are used explicitly
        self.rnn_size = config.pop("rnn_size")
        self.epochs = config.pop("epochs")
        self.batch_size = config.pop("batch_size")
        self.validation_split = config.pop("validation_split")
        self.random_seed = config.pop("random_seed")

        self._train_params = config

    @property
    def max_len(self):
        if self.model:
            return self.model.layers[0].batch_input_shape[1]
        else:
            return None

    @staticmethod
    def _fill_in_features_to_max_length(
        features: List[np.ndarray], max_history: int
    ) -> List[np.ndarray]:
        """
        Pad features with zeros to maximum length;
        Args:
            features: list of features for each dialog;
                each feature has shape [dialog_history x shape_attribute]
            max_history: maximum history of the dialogs
        Returns:
            padded features
        """
        feature_shape = features[0].shape[-1]
        features = [
            feature
            if feature.shape[0] == max_history
            else np.vstack(
                [np.zeros((max_history - feature.shape[0], feature_shape)), feature]
            )
            for feature in features
        ]
        return features

    def _get_features_for_attribute(self, attribute_data: Dict[Text, List[np.ndarray]]):
        """
        Given a list of all features for one attribute, turn it into a numpy array;
        shape_attribute = features[SENTENCE][0][0].shape[-1]
            (Shape of features of one attribute)
        Args:
            attribute_data: all features in the attribute stored in a np.array;
        Output:
            2D np.ndarray with features for an attribute with
                shape [num_dialogs x (max_history * shape_attribute)]
        """
        sentence_features = attribute_data[SENTENCE][0]
        if isinstance(sentence_features[0], scipy.sparse.coo_matrix):
            sentence_features = [feature.toarray() for feature in sentence_features]
        # MaxHistoryFeaturizer is always used with SkLearn policy;
        max_history = self.featurizer.max_history
        features = self._fill_in_features_to_max_length(sentence_features, max_history)
        features = [feature.reshape((1, -1)) for feature in features]
        return np.vstack(features)

    def _preprocess_data(self, data: Data) -> np.ndarray:
        """
        Turn data into np.ndarray for sklearn training; dialogue history features
        are flattened.
        Args:
            data: training data containing all the features
        Returns:
            Training_data: shape [num_dialogs x (max_history * all_features)];
            all_features - sum of number of features of
            intent, action_name, entities, forms, slots.
        """
        if TEXT in data or ACTION_TEXT in data:
            raise Exception(
                f"{self.__name__} cannot be applied to text data. "
                f"Try to use TEDPolicy instead. "
            )

        attribute_data = {
            attribute: self._get_features_for_attribute(attribute_data)
            for attribute, attribute_data in data.items()
        }
        # turning it into OrderedDict so that the order of features is the same
        attribute_data = OrderedDict(attribute_data)
        salida = np.concatenate(list(attribute_data.values()), axis=-1)
        #logger.info("VECTORIZATION 'X' ")
        #logger.info(salida)
        #logger.info(salida.shape)
        #logger.info("MAX_HISTORY")
        #max_history = self.featurizer.max_history
        #logger.info(max_history)
        #logger.info("salida.shape[0]")
        #logger.info(salida.shape[0])
        #logger.info("reshape")
        new_salida = salida.reshape((salida.shape[0], 5, -1))
        #logger.info("shape")
        #logger.info(new_salida.shape)
        #logger.info(new_salida)
        return new_salida

    def model_architecture(
        self, input_shape: Tuple[int, int], output_shape: Tuple[int, Optional[int]]
    ) -> tf.keras.models.Sequential:
        """ Se contruye el modelo BiLSTM - return a compiled model."""

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Masking,
            LSTM,
            Bidirectional,
            Dense,
            TimeDistributed,
            Activation,
            Dropout,
        )

        # Build Model
        model = Sequential()

        # the shape of the y vector of the labels,
        # determines which output from rnn will be used
        # to calculate the loss
        if len(output_shape) == 1:
            # y is (num examples, num features) so
            # only the last output from the rnn is used to
            # calculate the loss
            logger.info("ENTRO A OUTPU_SHAPE 1")
            logger.info("INPUT SHAPE VALUE")
            logger.info(input_shape)
            logger.info("OUTPUt_SHAPe")
            logger.info(output_shape[-1])
            logger.info("RNN SIZE")
            logger.info(self.rnn_size)

            model.add(Masking(mask_value=-1, input_shape=input_shape))
            model.add(Bidirectional(LSTM(self.rnn_size, return_sequences=True)))
            # 20% de las neuronas serán ignoradas durante el training (20%xNodos = 10)
            # Para hacer menos probable el overfiting
            model.add(Dropout(rate=0.2))
            model.add(Bidirectional(LSTM(self.rnn_size)))
            model.add(Dropout(rate=0.2))
            model.add(Dense(input_dim=self.rnn_size, units=output_shape[-1]))


        elif len(output_shape) == 2:
            # y is (num examples, max_dialogue_len, num features) so
            # all the outputs from the rnn are used to
            # calculate the loss, therefore a sequence is returned and
            # time distributed layer is used

            # the first value in input_shape is max dialogue_len,
            # it is set to None, to allow dynamic_rnn creation
            # during prediction
            logger.info("ENTRO A OUTPU_SHAPE 2")
            logger.info("INPUT SHAPE VALUE")
            logger.info(input_shape)
            logger.info("OUTPUt_SHAPe")
            logger.info(output_shape[-1])
            logger.info("RNN SIZE")
            logger.info(self.rnn_size)
            model.add(Masking(mask_value=-1, input_shape=(None, input_shape[1])))
            model.add(LSTM(self.rnn_size, return_sequences=True, dropout=0.2))
            model.add(TimeDistributed(Dense(units=output_shape[-1])))
        else:
            raise ValueError(
                "Cannot construct the model because"
                "length of output_shape = {} "
                "should be 1 or 2."
                "".format(len(output_shape))
            )

        model.add(Activation("softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
        )

        if common_utils.obtain_verbosity() > 0:
            model.summary()

        #plot_model(model, 'BiLSTM_model.png', show_shapes=True)
        return model

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        #logger.info("training_trackers")
        #logger.info(training_trackers[0].slots)
        #logger.info(training_trackers[0].sender_id)
        tracker_state_features, label_ids = self.featurize_for_training(
            training_trackers, domain, interpreter, **kwargs
        )
        training_data, zero_state_features = model_data_utils.convert_to_data_format(
            tracker_state_features
        )
        X = self._preprocess_data(training_data)
        y = label_ids
        shuffled_X, shuffled_y = X,y
        logger.info("VECTORIZATION y")
        logger.info(shuffled_y)
        logger.info("X")
        logger.info(shuffled_X.shape)
        logger.info(shuffled_X.shape[1:])

        logger.info("y")

        logger.info(shuffled_y.shape)
        logger.info(shuffled_y.shape[1:])
        logger.info(len(shuffled_y.shape[1:]))
        if self.model is None:
            self.model = self.model_architecture(
                shuffled_X.shape[1:], shuffled_y.shape[1:]
            )

        logger.debug(
            f"Fitting model with {len(shuffled_y)} total samples and a "
            f"validation split of {self.validation_split}."
        )

        # filter out kwargs that cannot be passed to fit
        self._train_params = self._get_valid_params(
            self.model.fit, **self._train_params
        )

        self.model.fit(
            shuffled_X,
            shuffled_y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=common_utils.obtain_verbosity(),
            **self._train_params,
        )
        self.current_epoch = self.epochs

        logger.debug("Done fitting Keras Policy model.")

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:

        X = self.featurizer.create_state_features([tracker], domain, interpreter)
        training_data, _ = model_data_utils.convert_to_data_format(
            X, self.zero_state_features
        )
        Xt = self._preprocess_data(training_data)
        y_pred = self.model.predict(Xt, batch_size=1)
        #logger.info("Y_PRED")
        #logger.info(y_pred)
        #logger.info("Y_PRED.SHAPE")
        #logger.info(y_pred.shape)
        if len(y_pred.shape) == 2:
            return y_pred[-1].tolist()
        elif len(y_pred.shape) == 3:
            return y_pred[0, -1].tolist()
        else:
            raise Exception("Network prediction has invalid shape.")

    def _postprocess_prediction(self, y_proba, domain) -> List[float]:
        yp = y_proba[0].tolist()

        # Some classes might not be part of the training labels. Since
        # sklearn does not predict labels it has never encountered
        # during training, it is necessary to insert missing classes.
        indices = self.label_encoder.inverse_transform(np.arange(len(yp)))
        y_filled = [0.0 for _ in range(domain.num_actions)]
        for i, pred in zip(indices, yp):
            y_filled[i] = pred

        return y_filled

    def persist(self, path: Union[Text, Path]) -> None:

        if self.model:
            self.featurizer.persist(path)

            meta = {
                "priority": self.priority,
                "model": "keras_model.h5",
                "epochs": self.current_epoch,
            }
            path = Path(path)

            #meta_file = os.path.join(path, "keras_policy.json")
            meta_file = path / "keras_policy.json"
            rasa.shared.utils.io.dump_obj_as_json_to_file(meta_file, meta)

            #model_file = os.path.join(path, meta["model"])
            model_file = path / meta["model"]

            # makes sure the model directory exists
            rasa.shared.utils.io.create_directory_for_file(model_file)
            self.model.save(model_file, overwrite=True)

        else:
            logger.debug(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )

    @classmethod
    def load(cls, path: Union[Text, Path]) -> Policy:
        from tensorflow.keras.models import load_model
        path = Path(path)
        filename = path / "keras_policy.json"

        if not Path(path).exists():
            raise OSError(
                f"Failed to load dialogue model. Path {filename.absolute()} "
                f"doesn't exist."
            )

        featurizer = TrackerFeaturizer.load(path)
        #meta_file = os.path.join(path, "lstm_policy.json")
        meta_file = path / "keras_policy.json"
        if os.path.isfile(meta_file):
            meta = json.loads(rasa.shared.utils.io.read_file(meta_file))

            #model_file = os.path.join(path, meta["model"])
            model_file = path / meta["model"]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = load_model(model_file)
            policy = cls(  # Constructor de la clase/construye esta
                featurizer=featurizer,
                priority=meta["priority"],
                model=model,
                current_epoch=meta["epochs"],
            )
        else:
            policy = cls(featurizer=featurizer)
        logger.info("Loaded LSTM model")
        return policy

# pytype: enable=import-error
# pytype: disable=module-attr
