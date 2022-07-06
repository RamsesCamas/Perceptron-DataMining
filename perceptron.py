class Perceptron:

    def __init__(self,inputs, weights, outputs=None, learning_rate=0) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights
        self.learning_rate = learning_rate
        self.threshold = 0

    def dot_product(self,input):
        y = 0
        for x_n, w_n in zip(input,self.weights):
            y += x_n * w_n
        return y

    def step_function(self, y):
        return 1 if y >= self.threshold else -1

    def train(self):
        contador = 0
        while contador < len(self.inputs):
            y = self.dot_product(self.inputs[contador])
            y = self.step_function(y)
            error = self.outputs[contador] - y
            if error == 0:
                contador += 1
            else:
                for i, w in enumerate(self.weights):
                    self.weights[i] = w + (self.learning_rate * error * self.inputs[contador][i])
                contador = 0
        return self.weights

    def predict(self,inputs):
        contador = 0
        while contador < len(inputs):
            y = self.dot_product(inputs[contador])
            y = self.step_function(y)
            if y == 1:
                print('Iris-setosa')
            else:
                print('Iris-versicolor')
            contador += 1