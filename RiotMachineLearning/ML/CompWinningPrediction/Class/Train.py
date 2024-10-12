import numpy as np

class Train:

    def __init__(self, model):
        self.model = model 

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        self.model.accuracy.init(y)

        train_steps = 1
        if validation_data is not None:
            X_val, y_val = validation_data
            validation_steps = len(X_val) // batch_size if batch_size else 1
        else:
            validation_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs + 1):
            print(f'Epoch: {epoch}')
            self.model.loss.new_pass()
            self.model.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                output = self.model.forward(batch_X, training=True)
                data_loss, regularization_loss = self.model.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                predictions = self.model.output_layer_activation.predictions(output)
                accuracy = self.model.accuracy.calculate(predictions, batch_y)

                self.model.backward(output, batch_y)
                self.model.optimizer.pre_update_params()
                for layer in self.model.trainable_layers:
                    self.model.optimizer.update_params(layer)
                self.model.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.model.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.model.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.model.accuracy.calculate_accumulated()

            print(f'Training: Acc: {epoch_accuracy:.3f}, Loss: {epoch_loss:.3f}')
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)

            if validation_data is not None:
                self._evaluate_validation(validation_data, batch_size, history)

        return history

    def predict(self, X, *, batch_size=None):
        predictions_steps = 1
        if batch_size is not None:
            predictions_steps = len(X) // batch_size
            if predictions_steps * batch_size < len(X):
                predictions_steps += 1

        output = []
        for step in range(predictions_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size: (step + 1) * batch_size]

            batch_output = self.model.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)

    def _evaluate_validation(self, validation_data, batch_size, history=None):
        X_val, y_val = validation_data
        validation_steps = 1

        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.model.loss.new_pass()
        self.model.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.model.forward(batch_X, training=False)
            self.model.loss.calculate(output, batch_y)

            predictions = self.model.output_layer_activation.predictions(output)
            self.model.accuracy.calculate(predictions, batch_y)

        validation_loss = self.model.loss.calculate_accumulated()
        validation_accuracy = self.model.accuracy.calculate_accumulated()

        print(f'Validation: Acc: {validation_accuracy:.3f}, Loss: {validation_loss:.3f}')

        if history:
            history['val_loss'].append(validation_loss)
            history['val_accuracy'].append(validation_accuracy)
