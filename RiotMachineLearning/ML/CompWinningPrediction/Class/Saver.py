import pickle
import copy

class Saver:
    @staticmethod
    def save_model(model, path):
        model_copy = copy.deepcopy(model)
        model_copy.loss.new_pass()
        model_copy.accuracy.new_pass()

        model_copy.input_layer.__dict__.pop('output', None)
        model_copy.loss.__dict__.pop('dinputs', None)

        for layer in model_copy.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model_copy, f)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def save_parameters(model, path):
        with open(path, 'wb') as f:
            pickle.dump(model.get_parameters(), f)

    @staticmethod
    def load_parameters(model, path):
        with open(path, 'rb') as f:
            model.set_parameters(pickle.load(f))
