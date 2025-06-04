class Smart {
  constructor(...layers) {
    this.layers = layers;
    this.weights = [];
    this.biases = [];

    for (let i = 0; i < layers.length - 1; i++) {
      this.weights.push(this.randomMatrix(layers[i], layers[i + 1]));
      this.biases.push(Array(layers[i + 1]).fill(0).map(() => Math.random()));
    }
  }

  randomMatrix(rows, cols) {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.random() * 2 - 1)
    );
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  dsigmoid(y) {
    return y * (1 - y);
  }

  forward(input) {
    let activations = [input];

    for (let i = 0; i < this.weights.length; i++) {
      const inputVector = activations[i];
      const weights = this.weights[i];
      const bias = this.biases[i];
      const output = [];

      for (let j = 0; j < bias.length; j++) {
        let sum = bias[j];
        for (let k = 0; k < inputVector.length; k++) {
          sum += inputVector[k] * weights[k][j];
        }
        output.push(this.sigmoid(sum));
      }

      activations.push(output);
    }

    return activations;
  }

  train(inputs, targets, learningRate = 0.1, epochs = 1000) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let idx = 0; idx < inputs.length; idx++) {
        const input = inputs[idx];
        const target = targets[idx];

        let activations = this.forward(input);
        let errors = [];

        let output = activations[activations.length - 1];
        let error = target.map((t, i) => t - output[i]);
        errors.unshift(error);

        for (let l = this.weights.length - 1; l > 0; l--) {
          let nextError = [];
          let currentWeights = this.weights[l];

          for (let i = 0; i < this.weights[l - 1][0].length; i++) {
            let sum = 0;
            for (let j = 0; j < currentWeights[0].length; j++) {
              sum += errors[0][j] * currentWeights[i][j];
            }
            nextError[i] = sum * this.dsigmoid(activations[l][i]);
          }

          errors.unshift(nextError);
        }

        for (let l = 0; l < this.weights.length; l++) {
          const layerInput = activations[l];
          const delta = errors[l];

          for (let i = 0; i < this.weights[l].length; i++) {
            for (let j = 0; j < this.weights[l][0].length; j++) {
              this.weights[l][i][j] += learningRate * delta[j] * layerInput[i];
            }
          }

          for (let j = 0; j < this.biases[l].length; j++) {
            this.biases[l][j] += learningRate * delta[j];
          }
        }
      }
    }
  }

  predict(input) {
    return this.forward(input).at(-1);
  }

  accuracy(testInputs, testTargets) {
    let correct = 0;
    for (let i = 0; i < testInputs.length; i++) {
      let output = this.predict(testInputs[i]);
      let predicted = output.map(x => Math.round(x));
      if (JSON.stringify(predicted) === JSON.stringify(testTargets[i])) {
        correct++;
      }
    }
    return (correct / testInputs.length) * 100;
  }

  normalizeArray(arr, max) {
    return arr.map(x => x / max);
  }
} 

function normalize(input, min, max) {
  return (input - min) / (max - min);
}

function denormalize(input, min, max) {
  return input * (max - min) + min;
}
