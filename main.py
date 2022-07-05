import random

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
        weight_n = random.uniform(0,1)
        initial_weights.append(weight_n)
    return initial_weights



if __name__ == '__main__':
    inputs, outputs = extract_data('iris.data')
    transformed_outputs = list(map(transform_outputs,outputs))
    learning_rate = random.uniform(0,1)
    initial_weights = generate_weights(inputs[0])
    print(initial_weights)