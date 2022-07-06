import random
from perceptron import Perceptron

def extract_data(filename):
    inputs = []
    outputs = []
    with open(filename,'r',encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            data = line.replace('\n','').split(',')
            input = data[0:4]
            output = data[4]
            input = list(map(float,input))
            inputs.append(input)
            outputs.append(output)
    return inputs, outputs

def transform_outputs(output):
    return 1 if output == 'Iris-setosa' else -1

def generate_weights(input):
    initial_weights = []
    for _ in range(len(input)):
        weight_n = random.uniform(-1,1)
        initial_weights.append(weight_n)
    return initial_weights

def split_data(inputs,outputs):
    setosa_input_data = inputs[0:len(inputs)//2] 
    setosa_output_data = outputs[0:len(inputs)//2] 
    versi_input_data = inputs[len(inputs)//2:len(inputs)] 
    versi_output_data = outputs[len(inputs)//2:len(inputs)] 

    test_data = setosa_input_data[0:10] + versi_input_data[0:10]
    train_input_data = setosa_input_data[10:len(setosa_input_data)] + versi_input_data[10:len(versi_output_data)]
    train_output_data = setosa_output_data[10:len(setosa_output_data)] + versi_output_data[10:len(versi_output_data)]

    return train_input_data, train_output_data, test_data

if __name__ == '__main__':
    inputs, outputs = extract_data('iris.data')
    train_input, train_output, test_data = split_data(inputs,outputs)
    transformed_outputs = list(map(transform_outputs,train_output))
    learning_rate = random.uniform(0,1)
    initial_weights = generate_weights(train_input[0])
    perceptron = Perceptron(train_input,initial_weights, transformed_outputs,learning_rate)
    final_weights = perceptron.train()
    perceptron_2 = Perceptron(test_data,final_weights)
    perceptron_2.predict(test_data)