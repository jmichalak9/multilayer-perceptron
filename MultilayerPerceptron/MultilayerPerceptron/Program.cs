// See https://aka.ms/new-console-template for more information

using MathNet.Numerics.LinearAlgebra;
using MultilayerPerceptron;

Layer[] layers = {new Layer(2), new Layer(1)};
var mlp = new MLP(layers, 0.1f);

var boolInputs = Matrix<Single>.Build.DenseOfArray(
    new float[,] {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
    });

var andLabels = Matrix<Single>.Build.DenseOfArray(
    new float[,] {
        {1},
        {0},
        {0},
        {0},
    });
var loss = mlp.Fit(100, boolInputs, andLabels);
Console.WriteLine(loss);